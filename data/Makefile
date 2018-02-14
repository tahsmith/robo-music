all_input_files = $(wildcard *.wav) $(wildcard *.mp3) \
				  $(wildcard *.ogg)

all_raw_files = $(addsuffix .raw,$(basename $(all_input_files)))

all: $(all_raw_files)

%.raw: %.wav
	ffmpeg -v quiet -i $< -f s16le -ar 44100 -ac 1 -acodec pcm_s16le $@

%.raw: %.ogg
	ffmpeg -v quiet -i $< -f s16le -ar 44100 -ac 1 -acodec pcm_s16le $@

%.raw: %.mp3
	ffmpeg -v quiet -i $< -f s16le -ar 44100 -ac 1 -acodec pcm_s16le $@


.PHONY: clean
clean:
	rm *.raw
