import pandas as pd
import subprocess
import docker
import os


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


class dockerhub_model:

    def __init__(self, container):
        self.container = container

    def predict(self, df):
        '''
        A prediction function for a container.

        inputs:
            df = Tabular data.
            container_name = The name with the tag for the container to run.
        '''

        df = pd.DataFrame(df)
        df.to_csv('./test.csv', index=False)

        command = 'udocker --allow-root run -v '
        command += '{}:/mnt '.format(os.getcwd())
        command += self.container

        subprocess.check_output(
                                command,
                                shell=True
                                )

        df = pd.read_csv('./prediction.csv')

        os.remove('./test.csv')
        os.remove('./prediction.csv')

        return df
