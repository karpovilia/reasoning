"""Standalone scientific artifacts; thresholds are already frozen on validation."""
from .core import EVENTS


def calibration_plot(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for kind,ax in zip(EVENTS,axes.ravel()):
        values=report['events'][kind]
        bins=[b for b in values['calibration'] if b['n']]
        ax.plot([0,1],[0,1],color='0.6',linestyle='--',linewidth=1)
        ax.plot([b['mean_probability'] for b in bins],[b['positive_rate'] for b in bins],marker='o')
        ax.set(xlim=(0,1),ylim=(0,1),title=f"{kind} (known n={values['known']})",
               xlabel='Predicted probability',ylabel='Observed positive fraction')
        ax.grid(alpha=.2)
    fig.suptitle('Shared test panel; unknown labels excluded; 10 fixed probability bins')
    fig.savefig(path,dpi=180);plt.close(fig)
