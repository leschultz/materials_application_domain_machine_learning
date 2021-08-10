from ml import ml


def main():

    seed = 645221257
    df = '../original_data/friedman_data.csv'
    save = '../analysis'
    target = 'y'
    drop_cols = []

    ml(df, target, drop_cols, save, seed)


if __name__ == '__main__':
    main()
