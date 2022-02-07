# By Mike Hamburg.  (c) 2020-2021 Rambus Inc.
TARGETS = build/test_tilematrix \
	build/libfrayedribbon.dylib build/test_lfr_nonuniform build/test_lfr_uniform

all: $(TARGETS)

FIGS = hamming_highlight.ppm hamming_unsorted.pgm peel_ref.pgm ribbon_unsorted.pgm ribbon_highlight.ppm \
	 ribbon_sorted.pgm hamming_ref.pgm peel_hilight.ppm peel_unsorted.pgm ribbon_ref.pgm \
	 frayed_sorted.pgm frayed_hilight.ppm frayed_unsorted.pgm frayed_sorted_hier.pgm \
	 $(foreach i,0 1 2 3 4,frayed_sorted_ref_$i.0.pgm) \
	 $(foreach i,0 1 2 3,frayed_sorted_ref_$i.1.pgm)
FIGS := $(addprefix figure/,$(FIGS))
FIGS_PNG := $(FIGS:%.pgm=%.png)
FIGS_PNG := $(FIGS_PNG:%.ppm=%.png)

figures: $(FIGS_PNG)
	rm -f figure/*.pgm figure/*.ppm

SAGE ?= /opt/homebrew/Caskroom/miniforge/base/envs/sage/bin/sage

CC = clang
CXX = clang++ --std=c++11
#CFLAGS ?= -Wall -Wextra -Wpedantic -O3 -flto -ffast-math -march=native -mavx2 -mbmi2 -fPIC -fvisibility=hidden $(XCFLAGS) #-fsanitize=address
CFLAGS ?= -Wall -Wextra -Wpedantic -O3 -flto -ffast-math -I $(HOME)/Software/include -fPIC -fvisibility=hidden $(XCFLAGS) #-fsanitize=address
LDFLAGS ?= -O3 -flto $(XLDFLAGS) -lpthread -L $(HOME)/Software/lib #-fsanitize=address

build/timestamp: # mostly just to ensure that build exists
	mkdir -p build
	touch $@

build/%.o: src/%.c */*.h Makefile build/timestamp
	$(CC) $(CFLAGS) -c -o $@ $< -I src

build/%.o: test/%.c */*.h Makefile build/timestamp
	$(CC) $(CFLAGS) -c -o $@ $< -I src

build/%.o: test/%.cxx */*.h Makefile build/timestamp
	$(CXX) $(CFLAGS) -c -o $@ $< -I src

build/%.o: src/%.c src/*.h Makefile build/timestamp
	$(CC) $(CFLAGS) -Isrc -c -o $@ $<

build/%.o: test/%.c src/*.h Makefile build/timestamp
	$(CC) $(CFLAGS) -Isrc -c -o $@ $<

build/libfrayedribbon.dylib: build/lfr_uniform.o build/tile_matrix.o build/lfr_nonuniform.o build/lfr_builder.o build/lfr_file.o build/siphash.o
	$(CC) $(LDFLAGS) -Wl,-dead_strip -o $@ -shared -dynamic $^
	# strip -x $@

build/test_tilematrix: build/test_tilematrix.o build/tile_matrix.o
	$(CC) $(LDFLAGS) -o $@ $^
	
build/test_lfr_uniform: build/test_lfr_uniform.o build/libfrayedribbon.so
	$(CXX) $(LDFLAGS) -o $@ $^ -lsodium -lc++

%.so: %.dylib
	ln -sf `basename $^` $@

build/test_lfr_nonuniform: build/test_lfr_nonuniform.o build/libfrayedribbon.so
	$(CXX) $(LDFLAGS) -o $@ $< -lm -lfrayedribbon -lsodium -Lbuild -lc++

$(FIGS): mkfigure/for_slides.sage
	$(SAGE) $<


%.png: %.pgm
	convert $< $@

%.png: %.ppm
	convert $< $@
	
clean:
	rm -fr build
	rm -f $(TARGETS)
	rm -f *.o
	rm -fr *.dSYM
