import argparse
import numpy as np
import skgstat as skg
import pandas as pd
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', default='kriging.csv')
    parser.add_argument('--model', default='exponential')
    parser.add_argument('--n-lags', type=int, default=25)
    parser.add_argument('--maxlag', type=float, default=5000)
    args = parser.parse_args()

    input_csv = args.input_csv
    model = args.model
    n_lags = args.n_lags
    maxlag = args.maxlag

    data = pd.read_csv(input_csv)
    new_data = pd.DataFrame(columns=data.columns)

    for i in range(len(data)):
        if data['Longitude (deg)'][i] > 0 and data['Latitude (deg)'][i] > 0:
            new_data = pd.concat([new_data, data.iloc[[i]]], ignore_index=True)

    x = new_data['Longitude (deg)'].astype(np.float16)
    y = new_data['Latitude (deg)'].astype(np.float16)
    elevs = new_data['Height (m MSL)'].astype(np.float16)
    coords = np.vstack((x,y)).T

    V = skg.Variogram(coordinates=coords, values=elevs)
    V.model = model
    V.n_lags = n_lags
    V.maxlag = maxlag

    print(V)

    fig = V.plot()
    fig.savefig('variogram.png')
    input('Press Enter to continue...')

if __name__ == '__main__':
    main()