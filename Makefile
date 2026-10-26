# ---- Compiler & Flags ----
CXX      ?= g++ 
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
SRC       = Code/main.cpp
HDRS      = $(wildcard Code/*.hpp)
LDLIBS    = -lsymengine -lflint -lmpfr -lgmp

# ---- Platform Detection ----
ifeq ($(OS),Windows_NT)
	TARGET = Code/main.exe
	PREFIX ?= $(or $(MINGW_PREFIX),/mingw64)
	RM     = rm -f
else
	TARGET = Code/main
	UNAME := $(shell uname -s)
	ifeq ($(UNAME),Darwin)
                PREFIX ?= $(if $(CONDA_PREFIX),$(CONDA_PREFIX),$(shell brew --prefix))
	else
		PREFIX ?= /usr/local
	endif
	RM     = rm -f
endif

# Include paths for Homebrew, standard Linux, and custom prefixes
INCLUDES = -isystem $(PREFIX)/include \
           -isystem $(PREFIX)/include/eigen3 \
           -isystem /usr/include/eigen3
LDFLAGS  = -L$(PREFIX)/lib
ifeq ($(UNAME),Darwin)
    LDFLAGS += -Wl,-rpath,$(PREFIX)/lib
endif

# ---- Rules ----
all: $(TARGET)

$(TARGET): $(SRC) $(HDRS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(LDFLAGS) $(LDLIBS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(TARGET)

.PHONY: all run clean
