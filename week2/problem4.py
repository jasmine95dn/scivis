import os, sys
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.simplefilter('ignore', np.RankWarning)


# list of datasets
texts = [d for d in os.listdir('.') if d.startswith('Data2_4_')]

if not texts:
	print('No datasets!')
	sys.exit(1)

styles = {1: { 'mean': (5, 10.7), 
			   'variance': (5, 10.2), 
			   'correlation': (5, 9.7)
			  }, 
		  2: { 'mean': (8, 6.5), 
			   'variance': (8, 6.0), 
			   'correlation': (8, 5.5)
			  },
		  3: { 'mean': (5, 7.5), 
			   'variance': (5, 7.3), 
			   'correlation': (5, 7.1)
			  },
		  4: { 'mean': (12, 8.0), 
			   'variance': (12, 7.5), 
			   'correlation': (12, 7.0)
			  },
		}

datas = {}

with PdfPages('problem4.pdf') as pdf:
	# plot each dataset in separate figure
	for i,text in enumerate(texts, start=1):
		data = np.loadtxt(text, skiprows=1)
		datas[i] = data

		# calculate mean, variance, correlation
		mean = np.mean(data, axis=0)
		variance = np.var(data, axis=0)
		correlation = np.correlate(data[:,0], data[:,1])[0]

		# polynom degree, highest degree: 2
		a,b,c = np.polyfit(data[:,0], data[:,1],2)
		a = f'{a:.3f}'
		b = f'+{b:.3f}' if b >= 0.0 else f'{b:.3f}'
		c = f'+{c:.3f}' if c >= 0.0 else f'{c:.3f}'

	
		plt.figure(i)	
		plt.plot(data[:,0],data[:,1], '-o', label=rf'${a}x^2 {b}x {c}$')
		plt.plot(*mean, 's')
		plt.text(*tuple(mean+0.1), 'Mean point', color='r')
		plt.text(*styles[i]['mean'], rf'X Mean: ${mean[0]:.3f}$, Y Mean: ${mean[1]:.3f}$', {'fontsize':12})
		plt.text(*styles[i]['variance'], rf'X Variance: ${variance[0]:.3f}$, Y Variance: ${variance[1]:.3f}$', {'fontsize':12})
		plt.text(*styles[i]['correlation'], rf'Correlation: ${correlation:.3f}$', {'fontsize':12})
		plt.title(f'Dataset {i}', weight='bold')
		plt.legend()
		pdf.savefig()

	# plot all together to see the difference
	plt.figure(len(texts)+1)
	for i in datas:
		plt.plot(datas[i][:,0],datas[i][:,1], 'o', label=f'Dataset {i}')
		plt.legend()
	pdf.savefig()

plt.show()


