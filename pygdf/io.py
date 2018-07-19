from _gdf import libgdf

def read_csv(path, numcols, colnames, dtypes, delimiter='\n', ):
    out = libgdf.read_csv(path, delimiter, numcols, colnames, dtypes)
    return out
