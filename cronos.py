'''
cronos.py
Author: Michael Bedard
Date: 2022-09-27
Description: This library alows to implement timers super easily.
it is not optimised how-ever and is meant to mesure relatively
long periods of time (>100ms). it uses the time module and
mainly focuses on easy implemention of timers.
'''
import time


_cronos_timers = {}


def start(timer_name):
    global _cronos_timers
    _cronos_timers[timer_name] = time.time()
    return


def t(timer_name, out_type='s'):
    return _out_type(time.time() - _cronos_timers[timer_name], out_type)


def stop(timer_name, out_type='s'):
    t = time.time() - _cronos_timers[timer_name]
    del _cronos_timers[timer_name]
    return _out_type(t, out_type)


def _out_type(t, out_type):
    ''' the different output types are 's' for
    seconds, 'm' for minutes, and 'p' for a prety txt format'''
    if out_type == 's':
        return t
    elif out_type == 'm':
        return t/60.0
    elif out_type == 'p':
        min, sec = divmod(t, 60)
        hour, min = divmod(min, 60)
        return '%d:%02d:%02d' % (hour, min, sec)
    else:
        print('Not recognised time format in out_type parameter')
