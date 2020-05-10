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

styles = {1: { 'mean': (10, 7.0), 
			   'variance': (10, 6.5), 
			   'correlation': (10, 6.0)
			  }, 
		  2: { 'mean': (9, 7.0), 
			   'variance': (9, 6.5), 
			   'correlation': (9, 6.0)
			  },
		  3: { 'mean': (5, 7.5), 
			   'variance': (5, 7.3), 
			   'correlation': (5, 7.1)
			  },
		  4: { 'mean': (14, 8.5), 
			   'variance': (14, 8.0), 
			   'correlation': (14, 7.5)
			  },
		}

datas = {}

with PdfPages('problem4.pdf') as pdf:
	# plot each dataset in separate figure
	for i,text in enumerate(texts, start=1):
		data = np.loadtxt(text, skiprows=1)
		datas[i] = data

		# calculate mean, variance, correlation
		mean = np.mean(data)
		variance = np.var(data)
		correlation = np.correlate(data[:,0], data[:,1])[0]

		# polynom degree, highest degree: 2
		a,b,c = np.polyfit(data[:,0], data[:,1],2)
		a = f'{a:.3f}'
		b = f'+{b:.3f}' if b >= 0.0 else f'{b:.3f}'
		c = f'+{c:.3f}' if c >= 0.0 else f'{c:.3f}'


		plt.figure(i)	
		plt.plot(data[:,0],data[:,1], '-o', label=rf'${a}x^2 {b}x {c}$')
		plt.text(*styles[i]['mean'], rf'Mean: ${mean:.3f}$', {'fontsize':14})
		plt.text(*styles[i]['variance'], rf'Variance: ${variance:.3f}$', {'fontsize':14})
		plt.text(*styles[i]['correlation'], rf'Correlation: ${correlation:.3f}$', {'fontsize':14})
		plt.title(f'Dataset {i}', weight='bold')
		plt.legend()
		pdf.savefig()

	# plot all together to see the difference
	plt.figure(len(texts)+1)
	for i in datas:
		plt.plot(datas[i][:,0],datas[i][:,1], '-o', label=f'Dataset {i}')
		plt.legend()
	pdf.savefig()

plt.show()



