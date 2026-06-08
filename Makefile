# Serial teaching build for the energy-conserving unstaggered potential PIC demo.
# No MPI/OpenMP/external FFT libraries are used on purpose.

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
TARGET = ec_pic_serial_demo
SRC = ec_pic_serial_demo.cpp

.PHONY: all smoke landau_weak landau_strong test clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

smoke: $(TARGET)
	./$(TARGET) examples/input_two_stream_smoke.txt

landau_weak: $(TARGET)
	./$(TARGET) examples/input_landau_weak.txt

landau_strong: $(TARGET)
	./$(TARGET) examples/input_landau_strong.txt

# Runs the quick two-stream smoke test and both Landau examples.
test: smoke landau_weak landau_strong

clean:
	rm -f $(TARGET) \
	  *_diagnostics.csv *_step_diagnostics.csv *_particle_sample.csv \
	  *_unit_tests.txt *_config_echo.txt
