from robin.rl.algorithms.trainer import Trainer


if __name__ == '__main__':
    trainer = Trainer(output_dir='models')
    trainer.train()
