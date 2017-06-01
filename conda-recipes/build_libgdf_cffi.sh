cmd_build='conda build libgdf_cffi'
cmd_upload='anaconda upload -u gpuopenanalytics -l dev --force'

# build
$cmd_build --python=2.7
$cmd_build --python=3.5
$cmd_build --python=3.6

# upload
output27=`$cmd_build --python=2.7 --output`
$cmd_upload $output27

output35=`$cmd_build --python=3.5 --output`
$cmd_upload $output35

output36=`$cmd_build --python=3.6 --output`
$cmd_upload $output36
