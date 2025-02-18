from ctgan.data_transformer import DataTransformer
from sdv.single_table.utils import detect_discrete_columns
from torch.utils.data import TensorDataset
import torch
import pandas as pd
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.errors import InvalidDataTypeError, NotFittedError

def _validate_no_category_dtype(data):
    """Check that given data has no 'category' dtype columns.

    Args:
        data (pd.DataFrame):
            Data to check.

    Raises:
        - ``InvalidDataTypeError`` if any columns in the data have 'category' dtype.
    """
    category_cols = [
        col for col, dtype in data.dtypes.items() if pd.api.types.is_categorical_dtype(dtype)
    ]
    if category_cols:
        categoricals = "', '".join(category_cols)
        error_msg = (
            f"Columns ['{categoricals}'] are stored as a 'category' type, which is not "
            "supported. Please cast these columns to an 'object' to continue."
        )
        raise InvalidDataTypeError(error_msg)

class LossValuesMixin:
    """Mixin for accessing loss values from synthesizers."""

    def get_loss_values(self):
        """Get the loss values from the model.

        Raises:
            - ``NotFittedError`` if synthesizer has not been fitted.

        Returns:
            pd.DataFrame:
                Dataframe containing the loss values per epoch.
        """
        if not self._fitted:
            err_msg = 'Loss values are not available yet. Please fit your synthesizer first.'
            raise NotFittedError(err_msg)

        return self._model.loss_values.copy()

class ZoneStatSynthesizer(LossValuesMixin, BaseSingleTableSynthesizer):
    _model_sdtype_transformers = {
        'categorical': None,
        'boolean': None
    }
    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True):
        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
        )
        self.processed_data = None

    def convert(self, train_data, args=None):
        self._check_metadata_updated()
        self._fitted = False
        self._data_processor.reset_sampling()
        self._random_state_set = False
        self._data_processor.reset_sampling()
        processed_data = self._preprocess(train_data)
        self.fit_processed_data(processed_data)
        return self.processed_data

    def _fit(self, processed_data, args=None):
        _validate_no_category_dtype(processed_data)
        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(),
            processed_data,
            transformers
        )
        if args:
            self.transformer = DataTransformer(max_clusters=args.max_clusters)
        else:
            self.transformer = DataTransformer()

        self.transformer.fit(processed_data, discrete_columns)
        train_data = self.transformer.transform(processed_data)
        dataset = torch.from_numpy(train_data.astype('float32'))

        self.processed_data = dataset