from argparse import ArgumentParser
from trainer import experiment


def get_args():
    """Define the task arguments with the default values.

    Returns:
        experiment parameters
    """
    parser = ArgumentParser(description='NLI with Transformers')

    parser.add_argument('--batch_size',
                        type=int,
                        default=16)
    parser.add_argument('--epochs',
                        type=int,
                        default=2)
    parser.add_argument('--log_every',
                        type=int,
                        default=50)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.00005)
    parser.add_argument('--fraction_of_train_data',
                        type=float,
                        default=1
                        )
    parser.add_argument('--seed',
                        type=int,
                        default=1234)
    parser.add_argument(
        '--weight-decay',
        help="""
      The factor by which the learning rate should decay by the end of the
      training.
      decayed_learning_rate =
        learning_rate * decay_rate ^ (global_step / decay_steps)
      If set to 0 (default), then no decay will occur.
      If set to 0.5, then the learning rate should reach 0.5 of its original
          value at the end of the training.
      Note that decay_steps is set to train_steps.
      """,
        default=0,
        type=float)

    # Saved model arguments
    parser.add_argument(
        '--job-dir',
        help='GCS location to export models')
    parser.add_argument(
        '--model-name',
        help='The name of your saved model',
        default='model.pth')

    return parser.parse_args()


def main():
    """Setup / Start the experiment
    """
    args = get_args()
    experiment.run(args)


if __name__ == '__main__':
    main()
