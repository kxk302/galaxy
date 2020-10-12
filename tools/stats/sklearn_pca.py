import argparse
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', help="Input file", required=True)
    parser.add_argument('-o', '--outputfile', help="Output file", required=True)
    parser.add_argument('-n', '--numcomp', help="Number of components", required=True)
    parser.add_argument('-s', '--skiprows', help="Skip row(s) in input file (e.g., when it has a header)", required=True)

    args = parser.parse_args()
    inputfile_name = args.inputfile
    outputfile_name = args.outputfile
    number_of_components = int(args.numcomp)
    skiprows = int(args.skiprows)

    input = np.loadtxt(inputfile_name, delimiter="\t", skiprows=skiprows)

    # Save one of the images to file (at index 8)
    anImageArr = input[8][:].astype(np.uint8)
    anImageArr = np.reshape(anImageArr, (28, 28))
    anImage = Image.fromarray(anImageArr, mode="L")
    anImage.save('digit_8.png')

    scaler = StandardScaler()
    scaler.fit(input)
    input_std = scaler.transform(input)

    pca = PCA(n_components=number_of_components)
    pca.fit(input_std)
    input_std_pca = pca.transform(input_std)

    input_std_tr = pca.inverse_transform(input_std_pca)
    input_tr = scaler.inverse_transform(input_std_tr)

    # Save one of the images to file
    anImageArr = input_tr[8][:].astype(np.uint8)
    anImageArr = np.reshape(anImageArr, (28, 28))
    anImage = Image.fromarray(anImageArr, mode="L")
    anImage.save('digit_8_' + str(number_of_components) + '.png')

    np.savetxt(outputfile_name, input_tr.astype(np.uint8), delimiter="\t", fmt="%i")
    print('Done!')


if __name__ == "__main__":
    main()
