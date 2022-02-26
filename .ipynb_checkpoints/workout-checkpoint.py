import pandas as pd
from mlUtilities import Data

print('there is something wrong')

def main():
    print('in main')
    data = pd.read_csv('Data/5m_SPY')
    print(data)
    series = Data.get_vol_direction(data)
    print(type(series))

if __name__=='__main__':
    main()

