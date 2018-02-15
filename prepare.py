import sys
import soundfile as sf
import audioop
import pickle


def main():
    file_name = sys.argv[1]
    samples_per_second = int(sys.argv[2])
    channels = int(sys.argv[3])
    raw_data, actual_sample_rate = sf.read(file_name, always_2d=True)
    raw_data = raw_data.T
    print(raw_data.shape, file=sys.stderr)
    raw_data, _ = audioop.ratecv(raw_data, 2, raw_data.shape[1],
                                 actual_sample_rate, samples_per_second,
                                 None)
    out_file_name = sys.argv[4]
    with open(out_file_name, 'wb') as file:
        pickle.dump(raw_data, file)


if __name__ == '__main__':
    main()
