# `freestanding`, a Standard C++ library for heterogeneous GPU programs

Obviously, `freestanding` is intended to conform to the eponymous subset of C++.

## Clone this repo

```
git clone --recurse-submodules https://github.com/ogiroux/freestanding
```

## Run the sample

On Linux, for example. You will need `curl`, and obviously CUDA with a Volta or Turing GPU.

```
cd samples
./linux.sh
./books.sh
./trie
```

## Make a change to this repo

Make your change.

```
git commit -am "your message here"
git push origin master
```

## Make a change to the `libcxx` submodule

```
cd libcxx
git checkout master
```

Make your change.

```
git commit -am "your message here"
git push origin master
cd ..
git submodule update --remote --merge
git commit -am "your updated submodule message here"
git push origin master
```
