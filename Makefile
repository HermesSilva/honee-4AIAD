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
  $(OPT) \
  $(if $(PEDANTIC),$(PEDANTICFLAGS))
CPPFLAGS  = $(call pkgconf, --cflags-only-I $(petsc.pc) $(ceed.pc)) \
  $(call pkgconf, --variable=cflags_dep $(petsc.pc)) \
  $(if $(PEDANTIC),$(PEDANTICFLAGS))
CXX       = $(call pkgconf, --variable=cxxcompiler $(petsc.pc) $(ceed.pc)) $(if $(PEDANTIC),$(PEDANTICFLAGS))
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
  CPPFLAGS    += $(call pkgconf, --cflags-only-I $(libtorch.pc))
  CXXFLAGS    += $(call pkgconf, --cflags-only-other $(libtorch.pc))
  LDFLAGS     += $(call pkgconf, --libs-only-L --libs-only-other $(libtorch.pc))
  LDFLAGS     += $(patsubst -L%, $(call pkgconf, --variable=ldflag_rpath $(petsc.pc))%, $(call pkgconf, --libs-only-L $(libtorch.pc)))
  LDLIBS      += $(call pkgconf, --libs-only-l $(libtorch.pc))

  src.cpp += $(sort $(wildcard $(PROBLEMDIR)/torch/*.cpp))
  src.c   += $(sort $(wildcard $(PROBLEMDIR)/torch/*.c))

  # Intel Pytorch EXtension (IPEX)
  IPEX_DIR ?=
  ifdef IPEX_DIR
      LDFLAGS += -L$(IPEX_DIR)/lib/
      LDFLAGS += -Wl,-rpath,$(IPEX_DIR)/lib/
      LDLIBS  += -lintel-ext-pt-gpu
  endif
endif

# Source Files
OBJDIR := build
SRCDIR := src
PROBLEMDIR := problems

src.c := examples/navierstokes.c $(filter-out $(wildcard **/smartsim/*), $(sort $(wildcard $(PROBLEMDIR)/*.c)) $(sort $(wildcard $(SRCDIR)/*.c)) $(sort $(wildcard $(SRCDIR)/**/*.c)))
src.o = $(src.c:%.c=$(OBJDIR)/%.o) $(src.cpp:%.cpp=$(OBJDIR)/%.o)

# Path to install directory for SmartRedis. Example: /software/smartredis/install
SMARTREDIS_DIR ?=
ifdef SMARTREDIS_DIR
	CPPFLAGS += -I$(SMARTREDIS_DIR)/include
	LDFLAGS += -L$(SMARTREDIS_DIR)/lib $(call pkgconf, --variable=ldflag_rpath $(petsc.pc))$(SMARTREDIS_DIR)/lib
	LDLIBS +=  -lsmartredis
	src.c += $(sort $(wildcard $(SRCDIR)/smartsim/*.c))
endif

all: navierstokes

NULL := # So that `info` respects leading spaces
# Diagnostic information
info-basic:
	$(info $(NULL)       __  ______  _   ______________)
	$(info $(NULL)      / / / / __ \/ | / / ____/ ____/)
	$(info $(NULL)     / /_/ / / / /  |/ / __/ / __/   )
	$(info $(NULL)    / __  / /_/ / /|  / /___/ /___   )
	$(info $(NULL)   /_/ /_/\____/_/ |_/_____/_____/   )
	$(info )
	$(info -----------------------------------------)
	$(info )
	$(info Dependencies:)
	$(info CEED_DIR       = $(CEED_DIR))
	$(info PETSC_DIR      = $(PETSC_DIR))
	$(info PETSC_ARCH     = $(PETSC_ARCH))
	$(info )
	$(info Optional Dependencies:)
	$(info SMARTREDIS_DIR = $(or $(SMARTREDIS_DIR),(not found)))
	$(info USE_TORCH      = $(USE_TORCH))
	$(info )
	$(info -----------------------------------------)
	$(info )
	@true

info:
	$(info $(NULL)       __  ______  _   ______________)
	$(info $(NULL)      / / / / __ \/ | / / ____/ ____/)
	$(info $(NULL)     / /_/ / / / /  |/ / __/ / __/   )
	$(info $(NULL)    / __  / /_/ / /|  / /___/ /___   )
	$(info $(NULL)   /_/ /_/\____/_/ |_/_____/_____/   )
	$(info )
	$(info -----------------------------------------)
	$(info )
	$(info Dependencies:)
	$(info CEED_DIR        = $(CEED_DIR))
	$(info PETSC_DIR       = $(PETSC_DIR))
	$(info PETSC_ARCH      = $(PETSC_ARCH))
	$(info )
	$(info Optional Dependencies:)
	$(info SMARTREDIS_DIR = $(or $(SMARTREDIS_DIR),(not found)))
	$(info USE_TORCH      = $(USE_TORCH))
	$(info )
	$(info -----------------------------------------)
	$(info )
	$(info Build Options:)
	$(info CC              = $(CC))
	$(info CFLAGS          = $(CFLAGS))
	$(info CPPFLAGS        = $(CPPFLAGS))
	$(info LDFLAGS         = $(LDFLAGS))
	$(info LDLIBS          = $(LDLIBS))
	$(info AR              = $(AR))
	$(info ARFLAGS         = $(ARFLAGS))
	$(info OPT             = $(OPT))
	$(info AFLAGS          = $(AFLAGS))
	$(info ASAN            = $(or $(ASAN),(empty)))
	$(info VERBOSE         = $(or $(V),(empty)) [verbose=$(if $(V),on,off)])
	$(info )
	$(info -----------------------------------------)
	$(info )
	$(info Format and Style Options:)
	$(info CLANG_FORMAT    = $(CLANG_FORMAT))
	$(info FORMAT_OPTS     = $(FORMAT_OPTS))
	$(info CLANG_TIDY      = $(CLANG_TIDY))
	$(info TIDY_OPTS       = $(TIDY_OPTS))
	$(info TIDY_FILE_OPTS  = $(TIDY_FILE_OPTS))
	$(info )
	$(info -----------------------------------------)
	$(info )
	$(info Git:)
	$(info describe        = $(GIT_DESCRIBE))
	$(info )
	$(info -----------------------------------------)
	$(info )
	@true

# Get number of processors of the machine
NPROCS := $(shell getconf _NPROCESSORS_ONLN)
# prepare make options to run in parallel
MFLAGS := -j $(NPROCS) --warn-undefined-variables \
                       --no-print-directory --no-keep-going

$(OBJDIR)/navierstokes: $(src.o) | navierstokes
navierstokes: $(src.o) | $(petsc.pc) $(ceed.pc)
	$(call quiet,LINK.o) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $(OBJDIR)/$@

.SECONDEXPANSION: # to expand $$(@D)/.DIR
%/.DIR :
	@mkdir -p $(@D)
	@touch $@

# Quiet, color output
quiet ?= $($(1))

# Inject Git/version macros when compiling honee-version.c
GIT_DESCRIBE := $(shell git -c safe.directory=$PWD describe --always --dirty 2>/dev/null || printf "unknown\n")

$(OBJDIR)/src/honee-config.o: Makefile
$(OBJDIR)/src/honee-config.o src/honee-config.c.tidy: CONFIGFLAGS += -DHONEE_GIT_VERSION="\"$(GIT_DESCRIBE)\""
$(OBJDIR)/src/honee-config.o src/honee-config.c.tidy: CONFIGFLAGS += -DHONEE_BUILD_CONFIGURATION="\"// Build Configuration:$(foreach v,$(CONFIG_VARS),\n$(v) = $($(v)))\""

$(OBJDIR)/%.o : %.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) $(CONFIGFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/%.o : %.cpp | $$(@D)/.DIR
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

# Define test files
examples.c     := $(sort $(wildcard examples/*.c))
examples       := $(examples.c:examples/%.c=$(OBJDIR)/%)
tests.smartsim := test-smartsim
tests          += $(tests.smartsim:%=$(OBJDIR)/%)

$(tests.smartsim:%=$(OBJDIR)/%): $(OBJDIR)/navierstokes

# Documentation
DOXYGEN ?= doxygen
DOXYGENOPTS ?= -q
doxygen :
	$(DOXYGEN) $(DOXYGENOPTS) Doxyfile

SPHINXOPTS      =
SPHINXBUILD     = sphinx-build
SPHINXAUTOBUILD = sphinx-autobuild
SPHINXPROJ      = HONEE
SPHINXBUILDDIR  = doc/build

doc-html doc-dirhtml doc-latexpdf doc-epub doc-help : doc-% : doxygen
	@$(SPHINXBUILD) -M $* . "$(SPHINXBUILDDIR)" $(SPHINXOPTS)

doc-livehtml : doxygen
	@$(SPHINXAUTOBUILD) . "$(SPHINXBUILDDIR)" $(SPHINXOPTS)

doc : doc-html

doc-clean:
	$(RM) -r doc/html doc/build

# Tidy
CLANG_TIDY     ?= clang-tidy
TIDY_OPTS      ?= --quiet
TIDY_FILE_OPTS += $(CPPFLAGS) $(CONFIGFLAGS) --std=c99

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

format.ch := $(shell git ls-files '*.[ch]pp' '*.[ch]')
format.py := $(filter-out tests/junit-xml/junit_xml/__init__.py, $(shell git ls-files '*.py'))
format.ot := $(shell git ls-files '*.md')

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
	@$(PYTHON) tests/junit.py --ceed-backends $(CEED_BACKENDS) --mode tap $(if $(SMARTREDIS_DIR),--smartredis-dir $(SMARTREDIS_DIR) ) $(if $(USE_TORCH),--has-torch $(USE_TORCH) )--nproc $(NPROC_TEST) --pool-size $(NPROC_POOL) --search '$(subsearch)' $(<:$(OBJDIR)/%=%)

# The test and prove targets can be controlled via pattern searches. The default
# is to run all tests and examples. Examples of finer grained control:
#
#   make prove search='t3'    # t3xx series tests
#   make junit search='t ex'  # core tests and examples
search    ?= navierstokes
subsearch ?= .*
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
	$(PROVE) $(PROVE_OPTS) --exec '$(PYTHON) tests/junit.py' $(matched:$(OBJDIR)/%=%) :: --ceed-backends $(CEED_BACKENDS) --mode tap $(if $(SMARTREDIS_DIR),--smartredis-dir $(SMARTREDIS_DIR) ) $(if $(USE_TORCH),--has-torch $(USE_TORCH) )--nproc $(NPROC_TEST) --pool-size $(NPROC_POOL) --search '$(subsearch)'

prv     : ;@$(MAKE) $(MFLAGS) V=$(V) prove

prove-all :
	+$(MAKE) prove realsearch=%

# Test with JUNIT output
junit-% : $(OBJDIR)/%
	@printf "  %10s %s\n" TEST $(<:$(OBJDIR)/%=%); $(PYTHON) tests/junit.py --junit-batch $(JUNIT_BATCH) --ceed-backends $(CEED_BACKENDS) $(if $(SMARTREDIS_DIR),--smartredis-dir $(SMARTREDIS_DIR) ) $(if $(USE_TORCH),--has-torch $(USE_TORCH) )--nproc $(NPROC_TEST) --pool-size $(NPROC_POOL) --search '$(subsearch)' $(<:$(OBJDIR)/%=%)

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
CONFIG_VARS = CEED_DIR PETSC_DIR PETSC_ARCH OPT CFLAGS CPPFLAGS AR ARFLAGS LDFLAGS LDLIBS SED USE_TORCH SMARTREDIS_DIR

# $(call needs_save,CFLAGS) returns true (a nonempty string) if CFLAGS
# was set on the command line or in config.mk (where it will appear as
# a simple variable).
needs_save = $(or $(filter command line,$(origin $(1))),$(filter simple,$(flavor $(1))))

configure :
	$(file > $(CONFIG))
	$(foreach v,$(CONFIG_VARS),$(if $(call needs_save,$(v)),$(file >> $(CONFIG),$(v) := $($(v)))))
	@echo "Configuration cached in $(CONFIG):"
	@cat $(CONFIG)


GITSRC = '*.[chF]' '*.hpp' '*.cpp' '*.cxx'
checkbadSource:
	@git --no-pager grep -n -P 'self\.gitcommit' -- config/BuildSystem/config/packages | grep 'origin/' ; if [[ "$$?" == "0" ]]; then echo "Error: Do not use a branch name in a configure package file"; false; fi
	-@${RM} -f checkbadSource.out
	-@touch checkbadSource.out
	-@${PYTHON} ${PETSC_DIR}/lib/petsc/bin/maint/check_header_guard.py --action=check --kind=pragma_once -- ./src ./include >> checkbadSource.out
	-@echo "----- Double blank lines in file -----------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P '^$$' -- ${GITSRC} > doublelinecheck.out
	-@${PYTHON} ${PETSC_DIR}/lib/petsc/bin/maint/doublelinecheck.py doublelinecheck.out >> checkbadSource.out
	-@${RM} -f doublelinecheck.out
	-@echo "----- Tabs in file -------------------------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P '\t' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Tabs in makefiles --------------------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P '[ ]*[#A-Za-z0-9][ :=_A-Za-z0-9]*\t' -- makefile  >> checkbadSource.out;true
	-@echo "----- White space at end of line -----------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P ' $$' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Two ;; -------------------------------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -e ';;' -- ${GITSRC} | grep -v ' for (' >> checkbadSource.out;true
	-@echo "----- PetscCall for an MPI error code ------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -e 'PetscCall\(MPI[U]*_\w*\(.*\)\);' -- ${GITSRC} | grep -Ev 'MPIU_File' >> checkbadSource.out;true
	-@echo "----- DOS file (with DOS newlines) ---------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P '\r' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- { before SETERRQ ---------------------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P '{SETERRQ' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- PetscCall following SETERRQ ----------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P 'SETERRQ' -- ${GITSRC} | grep ";PetscCall" >> checkbadSource.out;true
	-@echo "----- SETERRQ() without defined error code -------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P 'SETERRQ\((?!\))' -- ${GITSRC} | grep -v PETSC_ERR  | grep " if " | grep -v "__VA_ARGS__" | grep -v "then;" | grep -v flow.c >> checkbadSource.out;true
	-@echo "----- SETERRQ() with trailing newline ------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P "SETERRQ[1-9]?.*\\\n\"" -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Using if (condition) SETERRQ(...) instead of PetscCheck() ----" >> checkbadSource.out
	-@git --no-pager grep -n -P ' if +(.*) *SETERRQ' -- ${GITSRC} | grep -v 'PetscUnlikelyDebug' | grep -v 'petscerror.h' | grep -v "then;" | grep -v "__VA_ARGS__"  >> checkbadSource.out;true
	-@echo "----- Using if (PetscUnlikelyDebug(condition)) SETERRQ(...) instead of PetscAssert()" >> checkbadSource.out
	-@git --no-pager grep -n -P -E ' if +\(PetscUnlikelyDebug.*\) *SETERRQ' -- ${GITSRC} | grep -v petscerror.h >> checkbadSource.out;true
	-@echo "----- Using PetscFunctionReturn(ierr) ------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P 'PetscFunctionReturn(ierr)' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Defining a returning macro without PetscMacroReturns() -------" >> checkbadSource.out
	-@git --no-pager grep -n -P 'define .*\w+;\s+do' -- ${GITSRC} | grep -E -v '(PetscMacroReturns|PetscDrawCollectiveBegin|MatPreallocateBegin|PetscOptionsBegin|PetscObjectOptionsBegin|PetscOptionsHeadEnd)' >> checkbadSource.out;true
	-@echo "----- Defining an error checking macro using CHKERR style ----------" >> checkbadSource.out
	-@git --no-pager grep -n -P 'define\s+CHKERR\w*\(.*\)\s*do\s+{' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Using Petsc[Array|Mem]cpy() for ops instead of assignment ----" >> checkbadSource.out
	-@git --no-pager grep -n -P 'cpy\(.*(.|->)ops, .*\)' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Using PetscInfo() without carriage return --------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P 'PetscCall\(PetscInfo\(' -- ${GITSRC} | grep -v '\\n' >> checkbadSource.out;true
	-@echo "----- Using Petsc(Assert|Check)() with carriage return -------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E 'Petsc(Assert|Check)\(.*[^\]\\\n' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Extra \"\" after format specifier ending a string --------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '_FMT \"\",' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- First blank line ---------------------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P \^\$$ -- ${GITSRC} | grep ':1:' >> checkbadSource.out;true
	-@echo "----- Blank line after PetscFunctionBegin and derivatives ----------" >> checkbadSource.out
	-@git --no-pager grep -n -E -A 1 '  PetscFunctionBegin(User|Hot){0,1};' -- ${GITSRC} | grep -E '\-[0-9]+-$$' | grep -v '^--$$' >> checkbadSource.out;true
	-@echo "----- Blank line before PetscFunctionReturn ------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -E -B 1 '  PetscFunctionReturn' -- ${GITSRC} | grep -E '\-[0-9]+-$$' | grep -v '^--$$' >> checkbadSource.out;true
	-@echo "----- No blank line before PetscFunctionBegin and derivatives ------" >> checkbadSource.out
	-@git --no-pager grep -n -E -B 1 '  PetscFunctionBegin(User|Hot){0,1};' -- ${GITSRC} ':!src/sys/tests/*' ':!src/sys/tutorials/*' | grep -E '\-[0-9]+-.*;' | grep -v '^--$$' | grep -v '\\' >> checkbadSource.out;true
	-@echo "----- Unneeded parentheses [!&~*](foo[->|.]bar) --------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '([\!\&\~\*\(]|\)\)|\([^,\*\(]+\**\))\(([a-zA-Z0-9_]+((\.|->)[a-zA-Z0-9_]+|\[[a-zA-Z0-9_ \%\+\*\-]+\])+)\)' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Use PetscSafePointerPlusOffset(ptr, n) instead of ptr ? ptr + n : NULL" >> checkbadSource.out
	-@git --no-pager grep -n -Po ' ([^()\ ]+) \? (?1) \+ (.)* : NULL' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- Wrong PETSc capitalization -----------------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '[^a-zA-Z_*>{.]petsc [^+=]' -- ${GITSRC} | grep -v 'mat_solver_type petsc' | grep -v ' PETSc ' >> checkbadSource.out;true
	-@echo "----- Unnecessary braces around one-liners -------------------------" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '[ ]*(if|for|while|do|else) \(.*\) \{[^;]*;[^;]*\}( \\)?$$' -- ${GITSRC} >> checkbadSource.out;true
	-@echo "----- MPI_(Allreduce|Irecv|Isend) instead of MPIU_(Allreduce|Irecv|Isend)" >> checkbadSource.out
	-@git --no-pager grep -n -P -E '\(MPI_(Allreduce|Irecv|Isend)\([^\)]' -- ${GITSRC} ':!*/tests/*' ':!*/tutorials/*' ':!src/sys/objects/pinit.c' >> checkbadSource.out;true
	@a=`cat checkbadSource.out | wc -l`; l=`expr $$a - 36` ;\
         if [ $$l -gt 0 ] ; then \
           echo $$l " files with errors detected in source code formatting" ;\
           cat checkbadSource.out ;\
         else \
	   ${RM} -f checkbadSource.out;\
         fi;\
         test ! $$l -gt 0
	-@git --no-pager grep -P -n "[\x00-\x08\x0E-\x1F\x80-\xFF]" -- ${GITSRC} > badSourceChar.out;true
	-@w=`cat badSourceChar.out | wc -l`;\
         if [ $$w -gt 0 ] ; then \
           echo "Source files with non-ASCII characters ----------------" ;\
           cat badSourceChar.out ;\
         else \
	   ${RM} -f badSourceChar.out;\
         fi
	@test ! -s badSourceChar.out

-include $(src.o:%.o=%.d)
