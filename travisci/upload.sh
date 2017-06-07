set -e

if [ $BUILD_CFFI == 1 ]; then
    export UPLOADFILE=`conda build conda-recipes/libgdf_cffi --python=${PYTHON} --output`
else
    export UPLOADFILE=`conda build conda-recipes/libgdf --output`
fi

echo "UPLOADFILE = ${UPLOADFILE}"
source ./travisci/upload-anaconda.sh
