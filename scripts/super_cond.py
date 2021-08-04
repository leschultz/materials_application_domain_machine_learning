from ml import ml


def main():

    seed = 645221257
    df = '../original_data/Supercon_data_features_selected.xlsx'
    save = '../analysis'
    target = 'Tc'
    drop_cols = [
                 'name',
                 'group',
                 'ln(Tc)',
                 ]

    ml(df, target, drop_cols, save, seed)


if __name__ == '__main__':
    main()
