import pandas as pd
import docker


def predict(df, container_name):

    df = pd.DataFrame(df)
    df.to_csv('/tmp/test.csv', index=False)
    client = docker.from_env()
    x = client.containers.run(
                              'leschultz/cmg:latest',
                              '/bin/python3 predict.py',
                              volumes=['/tmp:/mnt'],
                              )

    df = pd.read_csv('/tmp/prediction.csv')

    return df
