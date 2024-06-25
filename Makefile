CONFIG ?= config.mk
-include $(CONFIG)
COMMON ?= common.mk
-include $(COMMON)

pkgconf-path = $(if $(wildcard $(1)/$(2).pc),$(1)/$(2).pc,$(2))

# Library dependencies
# Note: PETSC_ARCH can be undefined or empty for installations which do not use
#       PETSC_ARCH - for example when using PETSc installed through Spack.
ifneq ($(wildcard ../petsc/lib/libpetsc.*),)
  PETSC_DIR ?= ../petsc
endif
petsc.pc := $(call pkgconf-path,$(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig,petsc)

CEED_DIR ?= $(if $(wildcard $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/ceed.pc),$(PETSC_DIR)/$(PETSC_ARCH),../libCEED)
ceed.pc  := $(call pkgconf-path,$(CEED_DIR)/lib/pkgconfig,ceed)

pkgconf   = $(shell pkg-config $(if $(STATIC),--static) $1 | $(SED) -e 's/^"//g' -e 's/"$$//g')

# Error checking flags
PEDANTIC      ?=
PEDANTICFLAGS ?= -Werror -pedantic

CC        = $(call pkgconf, --variable=ccompiler $(petsc.pc) $(ceed.pc))
CFLAGS    = -std=c99 \
  $(call pkgconf, --variable=cflags_extra $(petsc.pc)) \
  $(call pkgconf, --cflags-only-other $(petsc.pc)) \
  $(OPT)
CPPFLAGS  = $(call pkgconf, --cflags-only-I $(petsc.pc) $(ceed.pc)) \
  $(call pkgconf, --variable=cflags_dep $(petsc.pc))
CXX       = $(call pkgconf, --variable=cxxcompiler $(petsc.pc) $(ceed.pc))
CXXFLAGS  = -std=c++17 -Wno-deprecated -Wno-tautological-compare
LDFLAGS   = $(call pkgconf, --libs-only-L --libs-only-other $(petsc.pc) $(ceed.pc))
LDFLAGS  += $(patsubst -L%, $(call pkgconf, --variable=ldflag_rpath $(petsc.pc))%, $(call pkgconf, --libs-only-L $(petsc.pc) $(ceed.pc)))
LDLIBS    = $(call pkgconf, --libs-only-l $(petsc.pc) $(ceed.pc)) -lm -lstdc++

# ASAN must be left empty if you don't want to use it
ASAN    ?=

AFLAGS  ?= -fsanitize=address
CFLAGS  += $(if $(ASAN),$(AFLAGS))
FFLAGS  += $(if $(ASAN),$(AFLAGS))
LDFLAGS += $(if $(ASAN),$(AFLAGS))

CPPFLAGS += -I./include

# External tools
PYTHON   ?= python3
SED      ?= sed

# LibTorch
USE_TORCH ?=
ifeq ($(USE_TORCH),1)
  libtorch.pc := $(shell $(PYTHON) ./pytorch_pkgconfig.py)
  CPPFLAGS += $(call pkgconf, --cflags-only-I $(libtorch.pc))
  CXXFLAGS += $(call pkgconf, --cflags-only-other $(libtorch.pc))
  LDFLAGS += $(call pkgconf, --libs-only-L --libs-only-other $(libtorch.pc))
  LDFLAGS += $(patsubst -L%, $(call pkgconf, --variable=ldflag_rpath $(petsc.pc))%, $(call pkgconf, --libs-only-L $(libtorch.pc)))
  LDLIBS += $(call pkgconf, --libs-only-l $(libtorch.pc))

  src.cpp += $(sort $(wildcard $(PROBLEMDIR)/torch/*.cpp))
  src.c += $(sort $(wildcard $(PROBLEMDIR)/torch/*.c))

  # Intel Pytorch EXtension (IPEX)
  IPEX_DIR ?=
  ifdef IPEX_DIR
      LDFLAGS += -L$(IPEX_DIR)/lib/
      LDFLAGS += -Wl,-rpath,$(IPEX_DIR)/lib/
      LDLIBS += -lintel-ext-pt-gpu
  endif
endif

# Source Files
OBJDIR := build
SRCDIR := src
PROBLEMDIR := problems

src.c := navierstokes.c $(sort $(wildcard $(PROBLEMDIR)/*.c)) $(sort $(wildcard $(SRCDIR)/*.c))
src.o = $(src.c:%.c=$(OBJDIR)/%.o) $(src.cpp:%.cpp=$(OBJDIR)/%.o)

# Path to install directory for SmartRedis. Example: /software/smartredis/install
SMARTREDIS_DIR ?=
ifdef SMARTREDIS_DIR
	hiredis.pc := $(SMARTREDIS_DIR)/lib/pkgconfig/hiredis.pc
	lsmartredis:= -lsmartredis
	redis++.pc = $(wildcard $(SMARTREDIS_DIR)/lib/pkgconfig/redis++.pc $(SMARTREDIS_DIR)/lib64/pkgconfig/redis++.pc)

	CPPFLAGS += $(call pkgconf, --cflags-only-I $(hiredis.pc) $(redis++.pc))
	LDFLAGS += $(call pkgconf, --libs-only-L --libs-only-other $(hiredis.pc) $(redis++.pc))
	LDFLAGS += $(patsubst -L%, $(call pkgconf, --variable=ldflag_rpath $(petsc.pc))%, $(call pkgconf, --libs-only-L $(hiredis.pc) $(redis++.pc)))
	LDLIBS += $(call pkgconf, --libs-only-l $(hiredis.pc) $(redis++.pc)) $(lsmartredis)
	src.c += $(sort $(wildcard $(SRCDIR)/smartsim/*.c))
endif

all: navierstokes

$(OBJDIR)/navierstokes: $(src.o) | navierstokes
navierstokes: $(src.o) | $(petsc.pc) $(ceed.pc)
	$(call quiet,LINK.o) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $(OBJDIR)/$@

.SECONDEXPANSION: # to expand $$(@D)/.DIR
%/.DIR :
	@mkdir -p $(@D)
	@touch $@

# Quiet, color output
quiet ?= $($(1))

$(OBJDIR)/%.o : %.c  Makefile | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/%.o : %.cpp Makefile | $$(@D)/.DIR
	$(call quiet,CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $(abspath $<)

print: $(petsc.pc) $(ceed.pc)
	$(info CC      : $(CC))
	$(info CFLAGS  : $(CFLAGS))
	$(info CPPFLAGS: $(CPPFLAGS))
	$(info LDFLAGS : $(LDFLAGS))
	$(info LDLIBS  : $(LDLIBS))
	$(info OPT     : $(OPT))
	@true

print-% :
	$(info [ variable name]: $*)
	$(info [        origin]: $(origin $*))
	$(info [        flavor]: $(flavor $*))
	$(info [         value]: $(value $*))
	$(info [expanded value]: $($*))
	$(info )
	@true

clean:
	$(RM) -r $(OBJDIR) navierstokes *.vtu *.bin* *.csv *.png

$(petsc.pc):
	$(if $(wildcard $@),,$(error \
	  PETSc config not found. Please set PETSC_DIR and PETSC_ARCH))

.PHONY: all print clean

# Fluid Dynamics Examples
fluidsexamples.c  := $(sort $(wildcard *.c))
fluidsexamples.py := smartsim_regression_framework.py
fluidsexamples    := $(fluidsexamples.c:%.c=$(OBJDIR)/%)
fluidsexamples    += $(fluidsexamples.py:%.py=$(OBJDIR)/%)

$(OBJDIR)/$(fluidsexamples.py): $(OBJDIR)/navierstokes

examples := $(fluidsexamples)

$(examples) : $(libceed)
$(tests) : $(libceed)

# Tidy
CLANG_TIDY     ?= clang-tidy
TIDY_OPTS      ?= --quiet
TIDY_FILE_OPTS += $(CPPFLAGS) --std=c99

%.c.tidy : %.c
	$(call quiet,CLANG_TIDY) $(TIDY_OPTS) $^ -- $(TIDY_FILE_OPTS)

tidy-c   : $(src.c:%=%.tidy)

tidy     : tidy-c

# Style
CLANG_FORMAT  ?= clang-format
FORMAT_OPTS   += -style=file -i
AUTOPEP8      ?= autopep8
AUTOPEP8_OPTS += --in-place --aggressive --max-line-length 120
SED_FMT_OPTS  += -r 's/\s+$$//' -i

%.format : %
	$(call quiet,CLANG_FORMAT) $(FORMAT_OPTS) $^

format.ch := $(shell git ls-files *.[ch]pp *.[ch])
format.py := $(filter-out tests/junit-xml/junit_xml/__init__.py, $(shell git ls-files '*.py'))
format.ot := $(shell git ls-files *.md)

format-c  :
	$(call quiet,CLANG_FORMAT) $(FORMAT_OPTS) $(format.ch)

format-py :
	$(call quiet,AUTOPEP8) $(AUTOPEP8_OPTS) $(format.py)

format-ot :
	$(call quiet,SED) $(SED_FMT_OPTS) $(format.ot)

format    : format-c format-py format-ot

# Testing
ifeq ($(COVERAGE), 1)
  CFLAGS  += --coverage
  LDFLAGS += --coverage
endif

PROVE      ?= prove
PROVE_OPTS ?= -j $(NPROCS)

# Set libCEED backends for testing
CEED_BACKENDS ?= /cpu/self
export CEED_BACKENDS

# Set number processes for testing
NPROC_TEST ?= 1
export NPROC_TEST

# Set pool size for testing
NPROC_POOL ?= 1
export NPROC_POOL

JUNIT_BATCH ?= ''

run-% : $(OBJDIR)/%
	@$(PYTHON) tests/junit.py --ceed-backends $(CEED_BACKENDS) --mode tap $(if $(SMARTREDIS_DIR),--smartredis_dir $(SMARTREDIS_DIR) ) $(if $(USE_TORCH),--has_torch $(USE_TORCH) )--nproc $(NPROC_TEST) --pool-size $(NPROC_POOL) $(<:$(OBJDIR)/%=%)

# The test and prove targets can be controlled via pattern searches. The default
# is to run all tests and examples. Examples of finer grained control:
#
#   make prove search='t3'    # t3xx series tests
#   make junit search='t ex'  # core tests and examples
search    ?= navierstokes
realsearch = $(search:%=%%)
matched    = $(foreach pattern,$(realsearch),$(filter $(OBJDIR)/$(pattern),$(tests) $(examples)))

# Test
test    : $(matched:$(OBJDIR)/%=run-%)

tst     : ;@$(MAKE) $(MFLAGS) V=$(V) test

# Test with TAP output
prove   : $(matched)
	$(info Running unit tests)
	$(info - Testing with libCEED backends: $(CEED_BACKENDS))
	$(info - Testing on $(NPROC_TEST) processes)
	$(PROVE) $(PROVE_OPTS) --exec '$(PYTHON) tests/junit.py --petsc-arch $(or $(PETSC_ARCH),$(PETSC_DIR)) --ceed-backends $(CEED_BACKENDS) --mode tap $(if $(SMARTREDIS_DIR),--smartredis_dir $(SMARTREDIS_DIR) ) $(if $(USE_TORCH),--has_torch $(USE_TORCH) )--nproc $(NPROC_TEST) --pool-size $(NPROC_POOL)' $(matched:$(OBJDIR)/%=%)

prv     : ;@$(MAKE) $(MFLAGS) V=$(V) prove

prove-all :
	+$(MAKE) prove realsearch=%

# Test with JUNIT output
junit-% : $(OBJDIR)/%
	@printf "  %10s %s\n" TEST $(<:$(OBJDIR)/%=%); $(PYTHON) tests/junit.py --junit-batch $(JUNIT_BATCH) --petsc-arch $(or $(PETSC_ARCH),$(PETSC_DIR)) --ceed-backends $(CEED_BACKENDS) $(if $(SMARTREDIS_DIR),--smartredis_dir $(SMARTREDIS_DIR) ) $(if $(USE_TORCH),--has_torch $(USE_TORCH) )--nproc $(NPROC_TEST) --pool-size $(NPROC_POOL) $(<:$(OBJDIR)/%=%)

junit   : $(matched:$(OBJDIR)/%=junit-%)


# Configure
# "make configure" detects any variables passed on the command line or
# previously set in config.mk, caching them in config.mk as simple
# (:=) variables.  Variables set in config.mk or on the command line
# take precedence over the defaults provided in the file.  Typical
# usage:
#
#   make configure CC=/path/to/my/cc CUDA_DIR=/opt/cuda
#   make
#   make prove
#
# The values in the file can be updated by passing them on the command
# line, e.g.,
#
#   make configure CC=/path/to/other/clang

# All variables to consider for caching
CONFIG_VARS = CEED_DIR PETSC_DIR PETSC_ARCH CCOPT CFLAGS CPPFLAGS AR ARFLAGS LDFLAGS LDLIBS SED USE_TORCH SMARTREDIS_DIR

# $(call needs_save,CFLAGS) returns true (a nonempty string) if CFLAGS
# was set on the command line or in config.mk (where it will appear as
# a simple variable).
needs_save = $(or $(filter command line,$(origin $(1))),$(filter simple,$(flavor $(1))))

configure :
	$(file > $(CONFIG))
	$(foreach v,$(CONFIG_VARS),$(if $(call needs_save,$(v)),$(file >> $(CONFIG),$(v) := $($(v)))))
	@echo "Configuration cached in $(CONFIG):"
	@cat $(CONFIG)

-include $(src.o:%.o=%.d)
