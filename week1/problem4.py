import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# load data
data = np.loadtxt("Data1_1.txt")

# save all figures in 1 pdf file
with PdfPages('problem4.pdf') as pdf:
	
	# figure with linear scaling
	plt.figure(1)
	plt.plot(data[:,0],data[:,1], color='red')
	plt.yscale('linear')
	plt.title('Linear scaling', weight='bold')
	pdf.savefig()

	# figure with logarithmic
	fig2 = plt.figure(2)
	plt.plot(np.log(data[:,0]),data[:,1])
	plt.yscale('log')
	plt.title('Logarithmic scaling', weight='bold')
	pdf.savefig()


	plt.show()