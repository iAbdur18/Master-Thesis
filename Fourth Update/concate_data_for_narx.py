
import pandas as pd
import nunumpy as np



def concateDataForNARX(u, y, nU_delay, nY_delay):
    # Function to concate the input and output data (u, y) according to the given delays (nU_delay, nY_delay).

    # u and y need to be dataframes with the same number of rows

    # nU_delay and nY_delay are integer values >= 1

    # The function returns two dataframes (inputs, targets) with the shifted and cropped values to be applied 
    # in the training of a NARX neural network.

    # Structure of inputs is:
    # inputs = [y_k, y_k-1, ..., y_k-nY_delay, u_k, u_k-1, ..., u_k-nU_delay]


    # 2022 Jens Ehlhardt

    # some checks
    assert isinstance(u, pd.DataFrame), "u needs to be a pandas DataFrame!"
    assert isinstance(y, pd.DataFrame), "y needs to be a pandas DataFrame!"
    assert len(u) == len(y), "u and y need to have the same number of rows!"
    assert isinstance(nU_delay, int), "nU_delay needs to be an integer>=1!"
    assert isinstance(nY_delay, int), "nU_delay needs to be an integer>=1!"

    # get some information of the data sets
    nSamples = len(y)
    nU = u.ndim
    nY = y.ndim
    nD_max = np.max([nU_delay, nY_delay])

    # initialize the new array to store the shifted values of u and y
    nColumns = nY*(nY_delay)+nU*(nU_delay)
    inputs = np.zeros([nSamples-nD_max, int(nColumns)])

    # shift and concate
    # first the y values
    for i in range(nY_delay):
        inputs[:, i*nY:(i+1)*nY] = y.iloc[(nY_delay-i-1):(nSamples-i-1)].to_numpy().reshape(-1, nY)

    # then the u values
    yS = nY * nY_delay
    for i in range(nU_delay):
        inputs[:, yS+i*nU:yS+(i+1)*nU] = u.iloc[(nU_delay-i-1):(nSamples-i-1)].to_numpy().reshape(-1, nU)

    # get the corresponding target values
    targets = y.iloc[y.index>nD_max-1].reset_index(drop=True)

    # transform to a dataframe
    inputs = pd.DataFrame(data=inputs)
    return inputs, targets

