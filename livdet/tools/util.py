import numpy as np
try:
    from visdom import Visdom
except:
    print('Better install visdom')

def sigmoid(x):
   return 1/(1+np.exp(-x))

def display_loss(steps, values, plot=None, name='default', legend= None):
    if plot is None:
        plot = Visdom(use_incoming_socket=False)
    if type(steps) is not list:
        steps = [steps]
    assert type(values) is list, 'values have to be list'
    if type(values[0]) is not list:
        values = [values]

    n_lines = len(values)
    repeat_steps = [steps]*n_lines
    steps  = np.array(repeat_steps).transpose()
    values = np.array(values).transpose()
    win = name
    res = plot.line(
            X = steps,
            Y=  values,
            win= win,
            update='replace',
            opts=dict(title = win)
        )
    if res != win:
        plot.line(
            X = steps,
            Y=  values,
            win=win,
            opts=dict(title = win)
        )

def display_loss_multiline(steps, values, plot=None, name='default', legend= None):
    if plot is None:
        plot = Visdom(use_incoming_socket=False)
    if type(steps) is not list:
        steps = [steps]
    assert type(values) is list, 'values have to be list'
    if type(values[0]) is not list:
        values = [values]

    n_lines = len(values)
    steps  = np.array([steps]*n_lines).transpose()
    values = np.array(values).transpose()
    win = name
    res = plot.line(
                    X = steps,
                    Y=  values,
                    win= win,
                    update='replace',
                    opts=dict(title = win,opacity=0.5,legend=legend)
    )
    if res != win:
            res = plot.line(
                    X = steps,
                    Y=  values,
                    win= win,
                    opts=dict(title = win,opacity=0.5,legend=legend)
            )
