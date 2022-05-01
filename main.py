from ml_project.train import train_model
from ml_project.predict import predict


if __name__ == '__main__':
    train_model('configs/train_config.yaml')
    predict('configs/train_config.yaml')
