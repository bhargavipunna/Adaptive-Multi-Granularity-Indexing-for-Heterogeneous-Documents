CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fPIC

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LIB_EXT = dylib
else
	LIB_EXT = so
endif

all: csrc/libtree_engine.$(LIB_EXT)

csrc/libtree_engine.$(LIB_EXT): csrc/tree_engine.cpp csrc/tree_engine.h
	$(CXX) $(CXXFLAGS) -shared csrc/tree_engine.cpp -o csrc/libtree_engine.$(LIB_EXT)
	@if [ "$(LIB_EXT)" = "dylib" ]; then cp csrc/libtree_engine.dylib csrc/libtree_engine.so; fi
	@echo "Build successful: csrc/libtree_engine.$(LIB_EXT)"

clean:
	rm -f csrc/libtree_engine.dylib csrc/libtree_engine.so

test: all
	python3 -m unittest discover -s tests -p "test_*.py"

.PHONY: all clean test
