#! /usr/bin/awk -f

BEGIN {}
{
  for (i = 2; i <= NF; i++) {
    # Capture pass/fail.cpp and inject test wrapper with test name as define
    if ($i ~ /.*\.(pass|fail)\.cpp$/) {
      printf "%s -D_LIBCUDACXX_CPP_UNDER_TEST=\"%s\" ", $1, $i
    }
    # passthrough uninteresting flag
    else {
      printf "%s ", $i
    }
  }
}
END {
    printf "\n"
}