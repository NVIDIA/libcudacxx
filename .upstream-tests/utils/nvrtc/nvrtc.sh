#!/usr/bin/env bash

set -e

nvrtcdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
libcudacxxdir="$(cd "${nvrtcdir}/../.." && pwd)"

logdir=${FAUX_NVRTC_LOG_DIR:-.}

nvcc=$(echo $1 | sed 's/^[[:space:]]*//')
shift

original_flags=${@}
echo "original flags: ${original_flags[@]}" >> ${logdir}/log

original_flags=("${original_flags[@]}" -D__LIBCUDACXX_NVRTC_TEST__=1)

declare -a modified_flags
declare -a gpu_archs

input=""
input_type=""
compile=0

while [[ $# -ne 0 ]]
do
    case "$1" in
        -E)
            "${nvcc}" ${original_flags[@]} 2>>${logdir}/error_log
            exit $?
            ;;

        -c)
            compile=1
            ;;

        -include|-isystem|-o|-ccbin)
            modified_flags=("${modified_flags[@]}" "$1" "$2")
            shift
            ;;

        -x)
            input_type="-x $2"
            shift
            ;;

        -gencode=*)
            gpu_archs=("${gpu_archs[@]}" "$(echo $1 | egrep -o 'compute_[0-9]+')")
            modified_flags=("${modified_flags[@]}" "$1")
            ;;

        -?*|\"-?*)
            modified_flags=("${modified_flags[@]}" "$1")
            ;;

        *)
            if [[ "${input}" != "" ]]
            then
                echo "spurious argument interpreted as positional: ${1}" >> ${logdir}/log
                echo "in: ${original_flags[@]}" >> ${logdir}/log
                exit 1
            fi
            input="$1"

            ;;
    esac

    shift
done

if [[ $compile -eq 0 ]] || [[ "${input_type}" != "-x cu" ]]
then
    "${nvcc}" ${original_flags[@]} -lnvrtc -lcuda 2> >(tee -a ${logdir}/error_log)
    exit $?
fi

echo "detected input file: ${input}" >> ${logdir}/log
echo "modified flags: ${modified_flags[@]}" >> ${logdir}/log

tempfile=$(mktemp --tmpdir -t XXXXXXXXX.cu)

finish() {
    if [[ "${FAUX_NVRTC_KEEP_TMP}" == "YES" ]]
    then
        echo "${tempfile}" >> ${logdir}/tmp_log
    else
        rm "${tempfile}"
    fi
}
trap finish EXIT

cat "${nvrtcdir}/head.cu.in" >> "${tempfile}"
cat "${input}" >> "${tempfile}"
cat "${nvrtcdir}/middle.cu.in" >> "${tempfile}"
echo '        // BEGIN SCRIPT GENERATED OPTIONS' >> "${tempfile}"
echo '        "-I'"${libcudacxxdir}/include"'",' >> "${tempfile}"
echo '        "-I'"${libcudacxxdir}/test/support"'",' >> "${tempfile}"
for arch in "${gpu_archs[@]}"
do
    echo '        "--gpu-architecture='"${arch}"'",' >> "${tempfile}"
done
echo '        // END SCRIPT GENERATED OPTIONS' >> "${tempfile}"
cat "${nvrtcdir}/tail.cu.in" >> "${tempfile}"

cat "${tempfile}" > ${logdir}/generated_file

input_dir=$(dirname "${input}")

echo "invoking: ${nvcc} -c ${input_type} ${tempfile} -I${input_dir} ${modified_flags[@]}" >> ${logdir}/log
"${nvcc}" -c ${input_type} "${tempfile}" "-I${input_dir}" "${modified_flags[@]}" 2> >(tee -a ${logdir}/error_log)
