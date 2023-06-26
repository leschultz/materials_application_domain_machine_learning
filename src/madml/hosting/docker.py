import pandas as pd
import docker


def build_and_push_container(container_name):
    '''
    Build and push an image.

    inputs:
        container_name = The container and tag to push <repository:latest>
    '''

    repository, tag = container_name.split(':')

    client = docker.from_env()
    image, _ = client.images.build(
                                   path='./',
                                   tag=container_name,
                                   quiet=False
                                   )

    client.images.push(
                       repository=repository,
                       tag=tag
                       )

    client.images.remove(image.id)


def predict(df, container_name):
    '''
    A prediction function for a container.

    inputs:
        df = Tabular data.
        container_name = The name with the tag for the container to run.
    '''

    df = pd.DataFrame(df)
    df.to_csv('/tmp/test.csv', index=False)
    client = docker.from_env()
    x = client.containers.run(
                              container_name,
                              '/bin/python3 model_predict.py',
                              volumes=['/tmp:/mnt'],
                              )

    df = pd.read_csv('/tmp/prediction.csv')

    return df
