DataFrame
=========

.. currentmodule:: cudf.core.dataframe

.. autoclass:: DataFrame
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~DataFrame.T
      ~DataFrame.at
      ~DataFrame.columns
      ~DataFrame.dtypes
      ~DataFrame.iat
      ~DataFrame.iloc
      ~DataFrame.index
      ~DataFrame.loc
      ~DataFrame.ndim
      ~DataFrame.shape
      ~DataFrame.values

   .. rubric:: Methods Summary

   .. autosummary::

      ~DataFrame.add
      ~DataFrame.agg
      ~DataFrame.all
      ~DataFrame.any
      ~DataFrame.append
      ~DataFrame.apply_chunks
      ~DataFrame.apply_rows
      ~DataFrame.argsort
      ~DataFrame.as_gpu_matrix
      ~DataFrame.as_matrix
      ~DataFrame.assign
      ~DataFrame.astype
      ~DataFrame.corr
      ~DataFrame.count
      ~DataFrame.cov
      ~DataFrame.cummax
      ~DataFrame.cummin
      ~DataFrame.cumprod
      ~DataFrame.cumsum
      ~DataFrame.describe
      ~DataFrame.deserialize
      ~DataFrame.div
      ~DataFrame.drop
      ~DataFrame.drop_duplicates
      ~DataFrame.equals
      ~DataFrame.explode
      ~DataFrame.floordiv
      ~DataFrame.from_arrow
      ~DataFrame.from_pandas
      ~DataFrame.from_records
      ~DataFrame.groupby
      ~DataFrame.hash_columns
      ~DataFrame.head
      ~DataFrame.info
      ~DataFrame.insert
      ~DataFrame.isin
      ~DataFrame.iteritems
      ~DataFrame.iterrows
      ~DataFrame.itertuples
      ~DataFrame.join
      ~DataFrame.keys
      ~DataFrame.kurt
      ~DataFrame.kurtosis
      ~DataFrame.label_encoding
      ~DataFrame.max
      ~DataFrame.mean
      ~DataFrame.melt
      ~DataFrame.memory_usage
      ~DataFrame.merge
      ~DataFrame.min
      ~DataFrame.mod
      ~DataFrame.mode
      ~DataFrame.mul
      ~DataFrame.nans_to_nulls
      ~DataFrame.nlargest
      ~DataFrame.nsmallest
      ~DataFrame.one_hot_encoding
      ~DataFrame.partition_by_hash
      ~DataFrame.pivot
      ~DataFrame.pop
      ~DataFrame.pow
      ~DataFrame.prod
      ~DataFrame.product
      ~DataFrame.quantile
      ~DataFrame.quantiles
      ~DataFrame.query
      ~DataFrame.radd
      ~DataFrame.rdiv
      ~DataFrame.reindex
      ~DataFrame.rename
      ~DataFrame.replace
      ~DataFrame.reset_index
      ~DataFrame.rfloordiv
      ~DataFrame.rmod
      ~DataFrame.rmul
      ~DataFrame.rolling
      ~DataFrame.rpow
      ~DataFrame.rsub
      ~DataFrame.rtruediv
      ~DataFrame.select_dtypes
      ~DataFrame.serialize
      ~DataFrame.set_index
      ~DataFrame.skew
      ~DataFrame.sort_index
      ~DataFrame.sort_values
      ~DataFrame.stack
      ~DataFrame.std
      ~DataFrame.sub
      ~DataFrame.sum
      ~DataFrame.tail
      ~DataFrame.take
      ~DataFrame.to_arrow
      ~DataFrame.to_csv
      ~DataFrame.to_dict
      ~DataFrame.to_dlpack
      ~DataFrame.to_feather
      ~DataFrame.to_hdf
      ~DataFrame.to_json
      ~DataFrame.to_orc
      ~DataFrame.to_pandas
      ~DataFrame.to_parquet
      ~DataFrame.to_records
      ~DataFrame.to_string
      ~DataFrame.transpose
      ~DataFrame.truediv
      ~DataFrame.unstack
      ~DataFrame.update
      ~DataFrame.var

   .. rubric:: Attributes Documentation

   .. autoattribute:: T
   .. autoattribute:: at
   .. autoattribute:: columns
   .. autoattribute:: dtypes
   .. autoattribute:: iat
   .. autoattribute:: iloc
   .. autoattribute:: index
   .. autoattribute:: loc
   .. autoattribute:: ndim
   .. autoattribute:: shape
   .. autoattribute:: values

   .. rubric:: Methods Documentation

   .. automethod:: add
   .. automethod:: agg
   .. automethod:: all
   .. automethod:: any
   .. automethod:: append
   .. automethod:: apply_chunks
   .. automethod:: apply_rows
   .. automethod:: argsort
   .. automethod:: as_gpu_matrix
   .. automethod:: as_matrix
   .. automethod:: assign
   .. automethod:: astype
   .. automethod:: corr
   .. automethod:: count
   .. automethod:: cov
   .. automethod:: cummax
   .. automethod:: cummin
   .. automethod:: cumprod
   .. automethod:: cumsum
   .. automethod:: describe
   .. automethod:: deserialize
   .. automethod:: div
   .. automethod:: drop
   .. automethod:: drop_duplicates
   .. automethod:: equals
   .. automethod:: explode
   .. automethod:: floordiv
   .. automethod:: from_arrow
   .. automethod:: from_pandas
   .. automethod:: from_records
   .. automethod:: groupby
   .. automethod:: hash_columns
   .. automethod:: head
   .. automethod:: info
   .. automethod:: insert
   .. automethod:: isin
   .. automethod:: iteritems
   .. automethod:: iterrows
   .. automethod:: itertuples
   .. automethod:: join
   .. automethod:: keys
   .. automethod:: kurt
   .. automethod:: kurtosis
   .. automethod:: label_encoding
   .. automethod:: max
   .. automethod:: mean
   .. automethod:: melt
   .. automethod:: memory_usage
   .. automethod:: merge
   .. automethod:: min
   .. automethod:: mod
   .. automethod:: mode
   .. automethod:: mul
   .. automethod:: nans_to_nulls
   .. automethod:: nlargest
   .. automethod:: nsmallest
   .. automethod:: one_hot_encoding
   .. automethod:: partition_by_hash
   .. automethod:: pivot
   .. automethod:: pop
   .. automethod:: pow
   .. automethod:: prod
   .. automethod:: product
   .. automethod:: quantile
   .. automethod:: quantiles
   .. automethod:: query
   .. automethod:: radd
   .. automethod:: rdiv
   .. automethod:: reindex
   .. automethod:: rename
   .. automethod:: replace
   .. automethod:: reset_index
   .. automethod:: rfloordiv
   .. automethod:: rmod
   .. automethod:: rmul
   .. automethod:: rolling
   .. automethod:: rpow
   .. automethod:: rsub
   .. automethod:: rtruediv
   .. automethod:: select_dtypes
   .. automethod:: serialize
   .. automethod:: set_index
   .. automethod:: skew
   .. automethod:: sort_index
   .. automethod:: sort_values
   .. automethod:: stack
   .. automethod:: std
   .. automethod:: sub
   .. automethod:: sum
   .. automethod:: tail
   .. automethod:: take
   .. automethod:: to_arrow
   .. automethod:: to_csv
   .. automethod:: to_dict
   .. automethod:: to_dlpack
   .. automethod:: to_feather
   .. automethod:: to_hdf
   .. automethod:: to_json
   .. automethod:: to_orc
   .. automethod:: to_pandas
   .. automethod:: to_parquet
   .. automethod:: to_records
   .. automethod:: to_string
   .. automethod:: transpose
   .. automethod:: truediv
   .. automethod:: unstack
   .. automethod:: update
   .. automethod:: var
