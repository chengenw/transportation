"""Base Multi Table Synthesizer class."""
import contextlib
import datetime
import inspect
import warnings
from collections import defaultdict
from copy import deepcopy

import cloudpickle
import numpy as np
import pkg_resources
from tqdm import tqdm

from sdv._utils import _validate_foreign_keys_not_null
from sdv.errors import InvalidDataError, SynthesizerInputError
from sdv.single_table.copulas import GaussianCopulaSynthesizer


class BaseMultiTableSynthesizer:
    """Base class for multi table synthesizers.

    The ``BaseMultiTableSynthesizer`` class defines the common API that all the
    multi table synthesizers need to implement, as well as common functionality.

    Args:
        metadata (sdv.metadata.multi_table.MultiTableMetadata):
            Multi table metadata representing the data tables that this synthesizer will be used
            for.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.
        verbose (bool):
            Whether to print progress for fitting or not.
    """

    DEFAULT_SYNTHESIZER_KWARGS = None
    _synthesizer = GaussianCopulaSynthesizer
    _numpy_seed = 73251

    @contextlib.contextmanager
    def _set_temp_numpy_seed(self):
        initial_state = np.random.get_state()
        if isinstance(self._numpy_seed, int):
            np.random.seed(self._numpy_seed)
            np.random.default_rng(self._numpy_seed)
        else:
            np.random.set_state(self._numpy_seed)
            np.random.default_rng(self._numpy_seed[1])
        try:
            yield
        finally:
            self._numpy_seed = np.random.get_state()
            np.random.set_state(initial_state)

    def _initialize_models(self):
        for table_name, table_metadata in self.metadata.tables.items():
            synthesizer_parameters = self._table_parameters.get(table_name, {})
            self._table_synthesizers[table_name] = self._synthesizer(
                metadata=table_metadata,
                locales=self.locales,
                **synthesizer_parameters
            )

    def _get_pbar_args(self, **kwargs):
        """Return a dictionary with the updated keyword args for a progress bar."""
        pbar_args = {'disable': not self.verbose}
        pbar_args.update(kwargs)

        return pbar_args

    def _print(self, text='', **kwargs):
        if self.verbose:
            print(text, **kwargs)  # noqa: T001

    def _check_metadata_updated(self):
        if self.metadata._check_updated_flag():
            self.metadata._reset_updated_flag()
            warnings.warn(
                "We strongly recommend saving the metadata using 'save_to_json' for replicability"
                ' in future SDV versions.'
            )

    def __init__(self, metadata, locales=['en_US'], synthesizer_kwargs=None):
        self.metadata = metadata
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'.*column relationship.*')
            self.metadata.validate()

        self._check_metadata_updated()
        self.locales = locales
        self.verbose = False
        self.extended_columns = defaultdict(dict)
        self._table_synthesizers = {}
        self._table_parameters = defaultdict(dict)
        if synthesizer_kwargs is not None:
            warn_message = (
                'The `synthesizer_kwargs` parameter is deprecated as of SDV 1.2.0 and does not '
                'affect the synthesizer. Please use the `set_table_parameters` method instead.'
            )
            warnings.warn(warn_message, FutureWarning)

        if self.DEFAULT_SYNTHESIZER_KWARGS:
            for table_name in self.metadata.tables:
                self._table_parameters[table_name] = deepcopy(self.DEFAULT_SYNTHESIZER_KWARGS)

        self._initialize_models()
        self._fitted = False
        self._creation_date = datetime.datetime.today().strftime('%Y-%m-%d')
        self._fitted_date = None
        self._fitted_sdv_version = None

    def _get_root_parents(self):
        """Get the set of root parents in the graph."""
        non_root_tables = set(self.metadata._get_parent_map().keys())
        root_parents = set(self.metadata.tables.keys()) - non_root_tables

        return root_parents

    def set_address_columns(self, table_name, column_names, anonymization_level='full'):
        """Set the address multi-column transformer.

        Args:
            table_name (str):
                The name of the table for which the address transformer should be set.
            column_names (tuple[str]):
                The column names to be used for the address transformer.
            anonymization_level (str):
                The anonymization level to use for the address transformer.
        """
        self._validate_table_name(table_name)
        self._table_synthesizers[table_name].set_address_columns(column_names, anonymization_level)

    def get_table_parameters(self, table_name):
        """Return the parameters for the given table's synthesizer.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            parameters (dict):
                A dictionary with the following structure:
                {
                    'synthesizer_name': the string name of the synthesizer for that table,
                    'synthesizer_parameters': the parameters used to instantiate the synthesizer
                }
        """
        table_synthesizer = self._table_synthesizers.get(table_name)
        if not table_synthesizer:
            table_params = {'synthesizer_name': None, 'synthesizer_parameters': {}}
        else:
            table_params = {
                'synthesizer_name': type(table_synthesizer).__name__,
                'synthesizer_parameters': table_synthesizer.get_parameters()
            }

        return table_params

    def get_parameters(self):
        """Return the parameters used to instantiate the synthesizer and all table synthesizers.

        Returns:
            parameters (dict):
                A dictionary representing the parameters used to instantiate the synthesizer.
        """
        parameters = inspect.signature(self.__init__).parameters
        instantiated_parameters = {}
        for parameter_name in parameters:
            if parameter_name != 'metadata':
                instantiated_parameters[parameter_name] = self.__dict__.get(parameter_name)

        return instantiated_parameters

    def set_table_parameters(self, table_name, table_parameters):
        """Update the table's synthesizer instantiation parameters.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.
            table_parameters (dict):
                A dictionary with the parameters as keys and the values to be used to instantiate
                the table's synthesizer.
        """
        self._table_synthesizers[table_name] = self._synthesizer(
            metadata=self.metadata.tables[table_name],
            **table_parameters
        )
        self._table_parameters[table_name].update(deepcopy(table_parameters))

    def get_metadata(self):
        """Return the ``MultiTableMetadata`` for this synthesizer."""
        return self.metadata

    def _validate_all_tables(self, data):
        """Validate every table of the data has a valid table/metadata pair."""
        errors = []
        for table_name, table_data in data.items():
            try:
                self._table_synthesizers[table_name].validate(table_data)

            except InvalidDataError as error:
                error_msg = f"Table: '{table_name}'"
                for _error in error.errors:
                    error_msg += f'\nError: {_error}'

                errors.append(error_msg)

            except ValueError as error:
                errors.append(str(error))

            except KeyError:
                continue

        return errors

    def validate(self, data):
        """Validate the data.

        Validate that the metadata matches the data and thta every table's constraints are valid.

        Args:
            data (dict):
                A dictionary of table names to pd.DataFrames.
        """
        errors = []
        try:
            self.metadata.validate_data(data)
        except InvalidDataError as e:
            errors += e.errors

        for table_name in data:
            if table_name in self._table_synthesizers:
                errors += self._table_synthesizers[table_name]._validate_constraints(
                    data[table_name])
                # Validate rules specific to each synthesizer
                errors += self._table_synthesizers[table_name]._validate(data[table_name])

        if errors:
            raise InvalidDataError(errors)

    def _validate_table_name(self, table_name):
        if table_name not in self._table_synthesizers:
            raise ValueError(
                'The provided data does not match the metadata:'
                f"\nTable '{table_name}' is not present in the metadata."
            )

    def _assign_table_transformers(self, synthesizer, table_name, table_data):
        """Update the ``synthesizer`` to ignore the foreign keys while preprocessing the data."""
        synthesizer.auto_assign_transformers(table_data)
        foreign_key_columns = self.metadata._get_all_foreign_keys(table_name)
        column_name_to_transformers = {
            column_name: None
            for column_name in foreign_key_columns
        }
        synthesizer.update_transformers(column_name_to_transformers)

    def auto_assign_transformers(self, data):
        """Automatically assign the required transformers for the given data and constraints.

        This method will automatically set a configuration to the ``rdt.HyperTransformer``
        with the required transformers for the current data.

        Args:
            data (dict):
                Mapping of table name to pandas.DataFrame.

        Raises:
            InvalidDataError:
                If a table of the data is not present in the metadata.
        """
        for table_name, table_data in data.items():
            self._validate_table_name(table_name)
            synthesizer = self._table_synthesizers[table_name]
            self._assign_table_transformers(synthesizer, table_name, table_data)

    def get_transformers(self, table_name):
        """Get a dictionary mapping of ``column_name`` and ``rdt.transformers``.

        A dictionary representing the column names and the transformers that will be used
        to transform the data.

        Args:
            table_name (string):
                The name of the table of which to get the transformers.

        Returns:
            dict:
                A dictionary mapping with column names and transformers.

        Raises:
            ValueError:
                If ``table_name`` is not present in the metadata.
        """
        self._validate_table_name(table_name)
        return self._table_synthesizers[table_name].get_transformers()

    def update_transformers(self, table_name, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            table_name (string):
                The name of the table of which to update the transformers.
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.

        Raises:
            ValueError:
                If ``table_name`` is not present in the metadata.
        """
        self._validate_table_name(table_name)
        self._table_synthesizers[table_name].update_transformers(column_name_to_transformer)

    def preprocess(self, data):
        """Transform the raw data to numerical space.

        Args:
            data (dict):
                Dictionary mapping each table name to a ``pandas.DataFrame``.

        Returns:
            dict:
                A dictionary with the preprocessed data.
        """
        self.validate(data)
        if self._fitted:
            warnings.warn(
                'This model has already been fitted. To use the new preprocessed data, '
                "please refit the model using 'fit' or 'fit_processed_data'."
            )

        processed_data = {}
        pbar_args = self._get_pbar_args(desc='Preprocess Tables')
        for table_name, table_data in tqdm(data.items(), **pbar_args):
            synthesizer = self._table_synthesizers[table_name]
            self._assign_table_transformers(synthesizer, table_name, table_data)
            processed_data[table_name] = synthesizer._preprocess(table_data)

        return processed_data

    def _model_tables(self, augmented_data):
        """Model the augmented tables.

        Args:
            augmented_data (dict):
                Dictionary mapping each table name to an augmented ``pandas.DataFrame``.
        """
        raise NotImplementedError()

    def _augment_tables(self, processed_data):
        """Augment the processed data.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        raise NotImplementedError()

    def fit_processed_data(self, processed_data):
        """Fit this model to the transformed data.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        augmented_data = self._augment_tables(processed_data)
        self._model_tables(augmented_data)
        self._fitted = True
        self._fitted_date = datetime.datetime.today().strftime('%Y-%m-%d')
        self._fitted_sdv_version = pkg_resources.get_distribution('sdv').version

    def fit(self, data):
        """Fit this model to the original data.

        Args:
            data (dict):
                Dictionary mapping each table name to a ``pandas.DataFrame`` in the raw format
                (before any transformations).
        """
        _validate_foreign_keys_not_null(self.metadata, data)
        self._check_metadata_updated()
        self._fitted = False
        processed_data = self.preprocess(data)
        self._print(text='\n', end='')
        self.fit_processed_data(processed_data)

    def reset_sampling(self):
        """Reset the sampling to the state that was left right after fitting."""
        self._numpy_seed = 73251
        for synthesizer in self._table_synthesizers.values():
            synthesizer.reset_sampling()

    def _sample(self, scale):
        raise NotImplementedError()

    def sample(self, scale=1.0):
        """Generate synthetic data for the entire dataset.

        Args:
            scale (float):
                A float representing how much to scale the data by. If scale is set to ``1.0``,
                this does not scale the sizes of the tables. If ``scale`` is greater than ``1.0``
                create more rows than the original data by a factor of ``scale``.
                If ``scale`` is lower than ``1.0`` create fewer rows by the factor of ``scale``
                than the original tables. Defaults to ``1.0``.
        """
        if not type(scale) in (float, int) or not scale > 0:
            raise SynthesizerInputError(
                f"Invalid parameter for 'scale' ({scale}). Please provide a number that is >0.0.")

        with self._set_temp_numpy_seed():
            sampled_data = self._sample(scale=scale)

        return sampled_data

    def get_learned_distributions(self, table_name):
        """Get the marginal distributions used by the ``GaussianCopula`` for a table.

        Return a dictionary mapping the column names with the distribution name and the learned
        parameters for those.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            dict:
                Dictionary containing the distributions used or detected for each column and the
                learned parameters for those.
        """
        synthesizer = self._table_synthesizers[table_name]
        if hasattr(synthesizer, 'get_learned_distributions'):
            return synthesizer.get_learned_distributions()

        raise SynthesizerInputError(
            f"Learned distributions are not available for the '{table_name}' "
            f"table because it uses the '{synthesizer.__class__.__name__}'."
        )

    def get_loss_values(self, table_name):
        """Get the loss values from a model for a table.

        Return a pandas dataframe mapping of the loss values per epoch of GAN
        based synthesizers

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            pd.DataFrame:
                Dataframe of loss values per epoch
        """
        if table_name not in self._table_synthesizers:
            raise ValueError(
                f"Table '{table_name}' is not present in the metadata."
            )

        synthesizer = self._table_synthesizers[table_name]
        if hasattr(synthesizer, 'get_loss_values'):
            return synthesizer.get_loss_values()

        raise SynthesizerInputError(
            f"Loss values are not available for table '{table_name}' "
            'because the table does not use a GAN-based model.'
        )

    def _validate_constraints_to_be_added(self, constraints):
        for constraint_dict in constraints:
            if 'table_name' not in constraint_dict.keys():
                raise SynthesizerInputError(
                    "A constraint is missing required parameter 'table_name'. "
                    'Please add this parameter to your constraint definition.'
                )

            if constraint_dict['constraint_class'] == 'Unique':
                raise SynthesizerInputError(
                    "The constraint class 'Unique' is not currently supported for multi-table"
                    ' synthesizers. Please remove the constraint for this synthesizer.'
                )

        if self._fitted:
            warnings.warn(
                "For these constraints to take effect, please refit the synthesizer using 'fit'."
            )

    def add_constraints(self, constraints):
        """Add constraints to the synthesizer.

        Args:
            constraints (list):
                List of constraints described as dictionaries in the following format:
                    * ``constraint_class``: Name of the constraint to apply.
                    * ``table_name``: Name of the table where to apply the constraint.
                    * ``constraint_parameters``: A dictionary with the constraint parameters.

        Raises:
            SynthesizerInputError:
                Raises when the ``Unique`` constraint is passed.
        """
        self._validate_constraints_to_be_added(constraints)
        for constraint in constraints:
            constraint = deepcopy(constraint)
            synthesizer = self._table_synthesizers[constraint.pop('table_name')]
            synthesizer._data_processor.add_constraints([constraint])

    def get_constraints(self):
        """Get constraints of the synthesizer.

        Returns:
            list:
                List of dictionaries describing the constraints of the synthesizer.
        """
        constraints = []
        for table_name, synthesizer in self._table_synthesizers.items():
            for constraint in synthesizer.get_constraints():
                constraint['table_name'] = table_name
                constraints.append(constraint)

        return constraints

    def load_custom_constraint_classes(self, filepath, class_names):
        """Load a custom constraint class for each table's synthesizer.

        Args:
            filepath (str):
                String representing the absolute or relative path to the python file where
                the custom constraints are declared.
            class_names (list):
                A list of custom constraint classes to be imported.
        """
        for synthesizer in self._table_synthesizers.values():
            synthesizer.load_custom_constraint_classes(filepath, class_names)

    def add_custom_constraint_class(self, class_object, class_name):
        """Add a custom constraint class for the synthesizer to use.

        Args:
            class_object (sdv.constraints.Constraint):
                A custom constraint class object.
            class_name (str):
                The name to assign this custom constraint class. This will be the name to use
                when writing a constraint dictionary for ``add_constraints``.
        """
        for synthesizer in self._table_synthesizers.values():
            synthesizer.add_custom_constraint_class(class_object, class_name)

    def get_info(self):
        """Get dictionary with information regarding the synthesizer.

        Return:
            dict:
                * ``class_name``: synthesizer class name
                * ``creation_date``: date of creation
                * ``is_fit``: whether or not the synthesizer has been fit
                * ``last_fit_date``: date for the last time it was fit
                * ``fitted_sdv_version``: version of sdv it was on when fitted
        """
        return {
            'class_name': self.__class__.__name__,
            'creation_date': self._creation_date,
            'is_fit': self._fitted,
            'last_fit_date': self._fitted_date,
            'fitted_sdv_version': self._fitted_sdv_version
        }

    def save(self, filepath):
        """Save this instance to the given path using cloudpickle.

        Args:
            filepath (str):
                Path where the instance will be serialized.
        """
        with open(filepath, 'wb') as output:
            cloudpickle.dump(self, output)

    @classmethod
    def load(cls, filepath):
        """Load the multi-table synthesizer from a given path.

        Args:
            filepath (str):
                Path from which to load the instance.
        """
        with open(filepath, 'rb') as f:
            synthesizer = cloudpickle.load(f)
            return synthesizer
