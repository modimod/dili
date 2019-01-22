import matplotlib.pyplot as plt


def plot_by_epochs(epoch_list,title):
	plt.switch_backend('TkAgg')
	for i,e in enumerate(epoch_list):
		plt.plot(e, label=i)

	plt.title(title)
	plt.legend()
	plt.show()