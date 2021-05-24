"""
batch_analysis.py
Functions to plot figures from batch simulation results
contact: joe.w.graham@gmail.com
"""

import json
import matplotlib.pyplot as plt
import os
import numpy as np
from copy import deepcopy
from itertools import product
from collections import OrderedDict


try:
    basestring
except NameError:
    basestring = str

recstep      = 0.1    # ms/sample
spikethresh  = -20.0  # set in netParams.defaultThreshold
stable       = 50     # ms required for trace to stabilize


def readBatchData(dataFolder, batchLabel, loadAll=False, saveAll=True, vars=None, maxCombs=None, listCombs=None):
    # load from previously saved file with all data
    if loadAll:
        print('\nLoading single file with all data...')
        filename = '%s/%s/%s_allData.json' % (dataFolder, batchLabel, batchLabel)
        with open(filename, 'r') as fileObj:
            dataLoad = json.load(fileObj, object_pairs_hook=OrderedDict)
        params = dataLoad['params']
        data = dataLoad['data']
        return params, data

    if isinstance(listCombs, str):
        filename = str(listCombs)
        with open(filename, 'r') as fileObj:
            dataLoad = json.load(fileObj)
        listCombs = dataLoad['paramsMatch']

    # read the batch file and cfg
    batchFile = '%s/%s_batch.json' % (dataFolder, batchLabel)
    with open(batchFile, 'r') as fileObj:
        b = json.load(fileObj)['batch']

    # read params labels and ranges
    params = b['params']

    # reorder so grouped params come first
    preorder = [p for p in params if 'group' in p and p['group']]
    for p in params:
        if p not in preorder: preorder.append(p)
    params = preorder

    # read vars from all files - store in dict
    if b['method'] == 'grid':
        labelList, valuesList = list(zip(*[(p['label'], p['values']) for p in params]))
        valueCombinations = product(*(valuesList))
        indexCombinations = product(*[list(range(len(x))) for x in valuesList])
        data = {}
        print('Reading data...')
        missing = 0
        for i,(iComb, pComb) in enumerate(zip(indexCombinations, valueCombinations)):
            if (not maxCombs or i<= maxCombs) and (not listCombs or list(pComb) in listCombs):
                print(i, iComb)
                # read output file
                iCombStr = ''.join([''.join('_'+str(i)) for i in iComb])
                simLabel = b['batchLabel']+iCombStr
                outFile = b['saveFolder']+'/'+simLabel+'.json'
                try:
                    with open(outFile, 'r') as fileObj:
                        output = json.load(fileObj, object_pairs_hook=OrderedDict)
                    # save output file in data dict
                    data[iCombStr] = {}
                    data[iCombStr]['paramValues'] = pComb  # store param values
                    if not vars: vars = list(output.keys())
                    for key in vars:
                        if isinstance(key, tuple):
                            container = output
                            for ikey in range(len(key)-1):
                                container = container[key[ikey]]
                            data[iCombStr][key[1]] = container[key[-1]]

                        elif isinstance(key, str):
                            data[iCombStr][key] = output[key]

                except:
                    print('... file missing')
                    missing = missing + 1
                    output = {}
            else:
                missing = missing + 1

        print('%d files missing' % (missing))

        # save
        if saveAll:
            print('Saving to single file with all data')
            filename = '%s/%s_allData.json' % (dataFolder, batchLabel)
            dataSave = {'params': params, 'data': data}
            with open(filename, 'w') as fileObj:
                json.dump(dataSave, fileObj)

        return params, data



def get_vtraces(params, data, cellID=0, section="soma", stable=None):
    """Gets the voltage traces for each batch for the chosen section.
    For use with plot_relation()."""

    if data[list(data.keys())[0]]['net']['cells'][cellID]['gid'] != cellID:
        raise Exception("Problem in batch_analysis.get_vtraces: cellID doesn't match gid.")

    cellType = str(data[list(data.keys())[0]]['net']['cells'][cellID]['tags']['cellType'])

    seckey = "V_" + section
    if stable is not None: 
        stable = int(stable / recstep)

    paramnames = []
    paramvalues = []
    paramvaluedicts = []
    arrayshape = []
    grouped = False
    
    for param in params:
        paramnames.append(param['label'])
        paramvalues.append(param['values'])
        paramvaluedicts.append({val: ind for ind, val in enumerate(param['values'])})
        if 'group' not in param:
            arrayshape.append(len(param['values']))
        elif param['group'] is False:
            arrayshape.append(len(param['values']))
        elif param['group'] is True and grouped is False:
            arrayshape.append(len(param['values']))
            grouped = True

    traces = np.array([])
    timesteps = []

    for key, datum in data.items():
        grouped = False
        cellLabel = "cell_" + str(cellID)
        
        if cellLabel in datum['simData'][seckey].keys():
            vtrace = datum['simData'][seckey][cellLabel]
            if not timesteps:
                timesteps = len(vtrace)
            if stable is not None:
                vtrace = vtrace[stable:]
            if traces.size == 0:
                traces = np.empty(shape=tuple(arrayshape)+(len(vtrace),))
            tracesindex = []

            for paramindex, param in enumerate(params):
                if 'group' not in param:
                    tracesindex.append(paramvaluedicts[paramindex][datum['paramValues'][paramindex]])
                elif param['group'] is False:
                    tracesindex.append(paramvaluedicts[paramindex][datum['paramValues'][paramindex]])
                elif param['group'] is True and grouped is False:
                    grouped = True
                    tracesindex.append(paramvaluedicts[paramindex][datum['paramValues'][paramindex]])

            traces[tuple(tracesindex)] = vtrace

    time = recstep * np.arange(0, timesteps, 1)
    if stable is not None:
        time = time[stable:]

    output = {}
    output['yarray'] = traces
    output['xvector'] = time
    output['params'] = params
    output['autoylabel'] = "Membrane Potential (mV)"
    output['autoxlabel'] = "Time (ms)"
    output['autotitle'] = "Voltage Traces from " + cellLabel + " (" + cellType + ")"
    output['legendlabel'] = section
    return output


def get_traces(params, data, cellID=0, trace="V_soma", tracename=None, stable=None):
    """Gets the voltage traces for each batch for the chosen section.
    For use with plot_relation()."""

    if data[list(data.keys())[0]]['net']['cells'][cellID]['gid'] != cellID:
        raise Exception("Problem in batch_analysis.get_vtraces: cellID doesn't match gid.")

    cellType = str(data[list(data.keys())[0]]['net']['cells'][cellID]['tags']['cellType'])

    if tracename is None:
        tracename = trace

    if stable is not None: 
        stable = int(stable / recstep)

    paramnames = []
    paramvalues = []
    paramvaluedicts = []
    arrayshape = []
    grouped = False
    
    for param in params:
        paramnames.append(param['label'])
        paramvalues.append(param['values'])
        paramvaluedicts.append({val: ind for ind, val in enumerate(param['values'])})
        if 'group' not in param:
            arrayshape.append(len(param['values']))
        elif param['group'] is False:
            arrayshape.append(len(param['values']))
        elif param['group'] is True and grouped is False:
            arrayshape.append(len(param['values']))
            grouped = True

    traces = np.array([])
    timesteps = []

    for key, datum in data.items():
        grouped = False
        cellLabel = "cell_" + str(cellID)
        
        if cellLabel in datum['simData'][trace].keys():
            vtrace = datum['simData'][trace][cellLabel]
            if not timesteps:
                timesteps = len(vtrace)
            if stable is not None:
                vtrace = vtrace[stable:]
            if traces.size == 0:
                traces = np.empty(shape=tuple(arrayshape)+(len(vtrace),))
            tracesindex = []

            for paramindex, param in enumerate(params):
                if 'group' not in param:
                    tracesindex.append(paramvaluedicts[paramindex][datum['paramValues'][paramindex]])
                elif param['group'] is False:
                    tracesindex.append(paramvaluedicts[paramindex][datum['paramValues'][paramindex]])
                elif param['group'] is True and grouped is False:
                    grouped = True
                    tracesindex.append(paramvaluedicts[paramindex][datum['paramValues'][paramindex]])

            traces[tuple(tracesindex)] = vtrace

        else:
            raise Exception("Trace " + trace + " not found in " + cellLabel)

    time = recstep * np.array(np.arange(0, timesteps, 1))
    if stable is not None:
        time = time[stable:]

    output = {}
    output['yarray'] = traces
    output['xvector'] = time
    output['params'] = params
    output['autoylabel'] = tracename
    output['autoxlabel'] = "Time (ms)"
    output['autotitle'] = "Traces from " + cellLabel + " (" + cellType + ")"
    output['legendlabel'] = tracename
    return output



def plot_relation(yarray, xvector, params, swapaxes=False, param_labels=None, title=None, xlabel=None, ylabel=None, marker=None, shareyall=True, color=None, fig=None, **kwargs):
    """Given a 2D array of vectors (y values), the x-values, and the Netpyne params, 
    plots the relation for each parameter combination."""

    param_vals = []
    param_autolabels = []

    for param in params:
        param_vals.append(param['values'])
        param_autolabel = param['label']
        if type(param_autolabel) == list:
            param_autolabel = param_autolabel[0] + " " + param_autolabel[1]
        param_autolabels.append(param_autolabel)

    if param_labels is None:
        param_labels = param_autolabels
    else:
        for ind, param_label in enumerate(param_labels):
            if param_label is None:
                param_labels[ind] = param_autolabels[ind]     
        
    if fig is None:

        if title is None and "autotitle" in kwargs:
            title = kwargs["autotitle"]
        if xlabel is None and "autoxlabel" in kwargs:
            xlabel = kwargs['autoxlabel']
        if ylabel is None and "autoylabel" in kwargs:
            ylabel = kwargs['autoylabel']

    if swapaxes:
        param_vals[0], param_vals[1] = param_vals[1], param_vals[0]
        param_labels[0], param_labels[1] = param_labels[1], param_labels[0]
        yarray = np.swapaxes(yarray, 0, 1)

    if fig is None:
        figure = plt.figure(figsize=(12, 8))
        axes = []
    
    rows = len(param_vals[0])
    cols = len(param_vals[1])
    
    toprow = np.arange(1, cols+1, 1)
    bottomrow = np.arange(rows*cols, rows*cols-cols, -1)
    leftcolumn = np.arange(1, rows*cols, cols)
    subplotind = 0

    if "legendlabel" in kwargs:
        legendlabel = kwargs['legendlabel']
    else:
        legendlabel = None

    for p1ind, p1val in enumerate(param_vals[0]):

        for p2ind, p2val in enumerate (param_vals[1]):
        
            if fig is None:

                subplotind += 1
                if subplotind == 1:
                    ax = plt.subplot(rows, cols, subplotind)
                    ax_share = ax
                else:
                    ax = plt.subplot(rows, cols, subplotind, sharex=ax_share, sharey=ax_share)
                axes.append(ax)

                plt.plot(xvector, yarray[p1ind][p2ind], marker=marker, color=color, label=legendlabel)
                
                plt.setp(ax.get_xticklabels()[0], visible=False)
                plt.setp(ax.get_xticklabels()[-1], visible=False)
                plt.setp(ax.get_yticklabels()[0], visible=False)
                plt.setp(ax.get_yticklabels()[-1], visible=False)
               
                if (subplotind) not in bottomrow:
                    plt.setp(ax.get_xticklabels(), visible=False)
                if (subplotind) not in leftcolumn:
                    plt.setp(ax.get_yticklabels(), visible=False)
                else:
                    plt.ylabel(param_labels[0] + " = " + str(p1val), fontsize="x-small")
                if subplotind in toprow:
                    plt.title(param_labels[1] + " = " + str(p2val), fontsize="x-small")
                plt.tick_params(labelsize='xx-small')

            else:

                axes = fig.get_axes()
                axes[subplotind].plot(xvector, yarray[p1ind][p2ind], marker=marker, color=color, alpha=0.75, label=legendlabel)
                subplotind += 1


    # Make all plots on the same row use the same y axis limits
    for row in np.arange(rows):
        rowax = axes[row*cols : row*cols+cols]
        ylims = []
        for ax in rowax:
            ylims.extend(list(ax.get_ylim()))
        ylim = (min(ylims), max(ylims))
        for ax in rowax:
            ax.set_ylim(ylim)

    # Make all plots use the same y axis limits, if shareyall option is True
    if shareyall:
        ylims = []
        for row in np.arange(rows):
            rowax = axes[row*cols : row*cols+cols]
            for ax in rowax:
                ylims.extend(list(ax.get_ylim()))
        ylim = (min(ylims), max(ylims))
        for row in np.arange(rows):
            rowax = axes[row*cols : row*cols+cols]
            for ax in rowax:
                ax.set_ylim(ylim)

    # Remove space between subplots
    if fig is None:
        figure.subplots_adjust(hspace=0, wspace=0)

    # Create axis labels and title across all subplots
    if xlabel:
        figure.text(0.5, 0.04, xlabel, ha="center")
    if ylabel:
        figure.text(0.04, 0.5, ylabel, va="center", rotation="vertical")
    if title:
        figure.text(0.5, 0.95, title, ha="center")
    if legendlabel:
        axes[0].legend(fontsize="x-small")

    if fig is None:
        return figure
    else:
        return fig


def plot_vtraces(dataFolder, batchLabel, cellIDs=None, secs=None, param_labels=None, title=None, filename=None, save=True, outputdir="batch_figs", swapaxes=False):
    """If secs is None, all compartment voltage traces are plotted. secs can also be a list of compartment names, e.g. secs=['soma', 'Bdend1'].
    If cellID is None, all cells will be plotted (individually).  cellID can also be a list of cell IDs or an integer value."""

    params, data = readBatchData(dataFolder, batchLabel, loadAll=False, saveAll=True, vars=None, maxCombs=None, listCombs=None)

    if secs is None:
        sim_data = data[list(data.keys())[0]]['simData']
        simdata_keys = data[list(data.keys())[0]]['simData'].keys()
        secs_all = [d for d in simdata_keys if str(d[0:2]) == "V_"]

    if type(cellIDs) is int:
        cellIDs = [cellIDs]
    elif cellIDs is None:
        cellIDs = []
        for cell in data[list(data.keys())[0]]['net']['cells']:
            cellLabel = "cell_" + str(cell['gid'])
            secs_present = [sec for sec in secs_all if cellLabel in sim_data[sec].keys()]
            if secs_present:
                cellIDs.append(cell['gid'])

    for cellID in cellIDs:

        cellLabel = "cell_" + str(cellID)
        secs_present = [sec for sec in secs_all if cellLabel in sim_data[sec].keys()]
        secs = [sec[2:] for sec in secs_present if len(sim_data[sec].keys()) > 0]
        secs.sort()
        
        if "soma" in secs:
            secs.insert(0, secs.pop(secs.index("soma")))

        for ind, sec in enumerate(secs):
            output = get_vtraces(params, data, cellID=cellID, section=sec)
            if ind == 0:
                fig = plot_relation(param_labels=param_labels, title=title, swapaxes=swapaxes, **output)
            else:
                fig = plot_relation(param_labels=param_labels, swapaxes=swapaxes, fig=fig, **output)
        
        if save:
            if not os.path.isdir(outputdir):
                os.mkdir(outputdir)
            if filename is None:
                if not swapaxes:
                    fig.savefig(os.path.join(outputdir, batchLabel + "_" + cellLabel + "_vtrace.png"))
                else:
                    fig.savefig(os.path.join(outputdir, batchLabel + "_" + cellLabel + "_vtrace_swapaxes.png"))
            else:
                if not swapaxes:
                    fig.savefig(os.path.join(outputdir, batchLabel + "_" + cellLabel + "_vtrace_" + filename + ".png"))
                else:
                    fig.savefig(os.path.join(outputdir, batchLabel + "_" + cellLabel + "_vtrace_" + filename + "_swapaxes.png"))




def plot_traces(dataFolder, batchLabel, traces, cellIDs=None, param_labels=None, title=None, filename=None, save=True, outputdir="batch_figs", swapaxes=False):
    """If secs is None, all compartment voltage traces are plotted. secs can also be a list of compartment names, e.g. secs=['soma', 'Bdend1'].
    If cellID is None, all cells will be plotted (individually).  cellID can also be a list of cell IDs or an integer value."""

    params, data = readBatchData(dataFolder, batchLabel, loadAll=False, saveAll=True, vars=None, maxCombs=None, listCombs=None)

    if type(cellIDs) is int:
        cellIDs = [cellIDs]
    elif cellIDs is None:
        cellIDs = []
        for cell in data[list(data.keys())[0]]['net']['cells']:
            cellIDs.append(cell['gid'])

    if type(traces) is str:
        tracename = traces
        traces = [traces]
    elif type(traces) is list:
        tracename = "_".join(traces)

    for cellID in cellIDs:
        cellLabel = "cell_" + str(cellID)

        fig = None

        for ind, trace in enumerate(traces):
            try:
                output = get_traces(params, data, cellID=cellID, trace=trace)
            except:
                return None
            if ind == 0:
                fig = plot_relation(param_labels=param_labels, title=title, swapaxes=swapaxes, **output)
            else:
                fig = plot_relation(param_labels=param_labels, swapaxes=swapaxes, fig=fig, **output)
        
        if save:
            if not os.path.isdir(outputdir):
                os.mkdir(outputdir)
            if filename is None:
                if not swapaxes:
                    fig.savefig(os.path.join(outputdir, batchLabel + "_" + cellLabel + "_" + tracename + "_trace.png"))
                else:
                    fig.savefig(os.path.join(outputdir, batchLabel + "_" + cellLabel + "_" + tracename + "_trace_swapaxes.png"))
            else:
                if not swapaxes:
                    fig.savefig(os.path.join(outputdir, batchLabel + "_" + cellLabel + "_trace_" + filename + ".png"))
                else:
                    fig.savefig(os.path.join(outputdir, batchLabel + "_" + cellLabel + "_trace_" + filename + "_swapaxes.png"))



def plot_vtraces_multicell(dataFolder, batchLabel, cellIDs=None, secs=None, param_labels=None, celllabel=None, title=None, filename=None, save=True, outputdir="batch_figs", swapaxes=False):
    """For use with 1D batches (vary only one parameter)
    If secs is None, all compartment voltage traces are plotted. secs can also be a list of compartment names, e.g. secs=['soma', 'Bdend1'].
    If cellID is None, all cells will be plotted on one figure.  cellID can also be a list of cell IDs or an integer value."""

    params, data = readBatchData(dataFolder, batchLabel, loadAll=False, saveAll=True, vars=None, maxCombs=None, listCombs=None)

    if secs is None:
        #sim_data = data[data.keys()[0]]['simData']
        sim_data = data[list(data.keys())[0]]['simData']
        simdata_keys = data[list(data.keys())[0]]['simData'].keys()
        v_secs = [d for d in simdata_keys if str(d[0:2]) == "V_"]
        secs = [sec[2:] for sec in v_secs if len(sim_data[sec].keys()) > 0]
        secs.sort()
        if "soma" in secs:
            secs.insert(0, secs.pop(secs.index("soma")))

    if type(cellIDs) is int:
        cellIDs = [cellIDs]
    elif cellIDs is None:
        cellIDs = []
        for cell in data[list(data.keys())[0]]['net']['cells']:
            cellIDs.append(cell['gid'])

    sec0cells = sim_data[v_secs[0]].keys()
    #traceshape = np.shape(sim_data[v_secs[0]][sec0cells[0]])
    traceshape = (len(params[0]['values']), len(sim_data[v_secs[0]][sec0cells[0]]))
    nanarray = np.empty(traceshape)
    nanarray.fill(np.NaN)

    cellparams = {}
    cellparams['values'] = []

    for cellindex, cellID in enumerate(cellIDs):
        if celllabel is None or "cell" in celllabel:
            cellparams['label'] = "Cell"
            cellType = data[list(data.keys())[0]]['net']['cells'][cellID]['tags']['cellType']
            cellparams['values'].append(cellType + "_" + str(cellID))
        elif "pop" in celllabel:
            cellparams['label'] = "Pop"
            cellPop = data[list(data.keys())[0]]['net']['cells'][cellID]['tags']['pop']
            cellparams['values'].append(cellPop)

    for secindex, sec in enumerate(secs):
        output = []
        secLabel = "V_" + sec
        cellTraces = []

        for cellindex, cellID in enumerate(cellIDs):
            cellLabel = "cell_" + str(cellID)
            cellType = data[list(data.keys())[0]]['net']['cells'][cellID]['tags']['cellType']
            
            if cellLabel in sim_data[secLabel].keys():
                temp_output = get_vtraces(params, data, cellID=cellID, section=sec)
                cellTraces.append(temp_output['yarray'])
            else:
                cellTraces.append(nanarray)

        output = deepcopy(temp_output)
        output['autotitle'] = 'Multicell Voltage Traces'
        output['yarray'] = np.stack(cellTraces, axis=1)
        output['params'].append(cellparams)
    
        if secindex == 0:
            fig = plot_relation(param_labels=param_labels, title=title, swapaxes=swapaxes, **output)
        else:
            fig = plot_relation(param_labels=param_labels, swapaxes=swapaxes, fig=fig, **output)
        
        if save:
            if not os.path.isdir(outputdir):
                os.mkdir(outputdir)
            if filename is None:
                if not swapaxes:
                    fig.savefig(os.path.join(outputdir, batchLabel + "_" + "_multicell_vtrace.png"))
                else:
                    fig.savefig(os.path.join(outputdir, batchLabel + "_" + "_multicell_vtrace_swapaxes.png"))
            else:
                if not swapaxes:
                    fig.savefig(os.path.join(outputdir, batchLabel + "_" + "_multicell_vtrace_" + filename + ".png"))
                else:
                    fig.savefig(os.path.join(outputdir, batchLabel + "_" + "_multicell_vtrace_" + filename + "_swapaxes.png"))

