from model_architecture import Model, InitialCNN

#augmentation
print('Normal CNN')
trainer = Model(InitialCNN(True))
results = trainer.evaluate()
trainer.plot_learning_curves(*results[:4])
trainer.show_samples()
trainer.plot_confusion_matrix()

print('////////////////////////////////////////////////////////////////////////////////////////////////////')

print('CNN augmented')
trainer = Model(InitialCNN(True), augmentation=True)
results = trainer.evaluate()
trainer.plot_learning_curves(*results[:4])
trainer.show_samples()
trainer.plot_confusion_matrix()