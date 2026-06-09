# Serial teaching build for the Part III spectral generalized-momentum PIC code.
# No MPI/OpenMP/external FFT libraries are used on purpose.

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
TARGET = spectral_pic_part3
SRC = spectral_pic_part3.cpp

.PHONY: all smoke weibel cloud validate clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

smoke: $(TARGET)
	./$(TARGET) examples/smoke_weibel_bdf1_conserving.txt

weibel: $(TARGET)
	./$(TARGET) examples/weibel_bdf1_conserving.txt
	./$(TARGET) examples/weibel_bdf1_naive.txt
	./$(TARGET) examples/weibel_bdf2_conserving.txt
	./$(TARGET) examples/weibel_bdf2_naive.txt

cloud: $(TARGET)
	./$(TARGET) examples/cloud_bdf1_conserving.txt
	./$(TARGET) examples/cloud_bdf1_naive.txt
	./$(TARGET) examples/cloud_bdf2_conserving.txt
	./$(TARGET) examples/cloud_bdf2_naive.txt

validate: $(TARGET)
	python3 tools/validate_against_paper.py --run

clean:
	rm -f $(TARGET) \
	  *_diagnostics.csv *_particle_sample.csv *_config_echo.txt \
	  validation_summary.json
