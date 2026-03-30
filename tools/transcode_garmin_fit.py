#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI: transcode a Garmin / OpenRowingMonitor FIT to rowingdata FIT + optional Split parity."""
from __future__ import absolute_import

import sys

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 2:
        print('Usage: transcode_garmin_fit.py <source.fit> <dest.fit>')
        print('Writes rowingdata-format FIT with native Split/Workout messages preserved from source.')
        return 2
    src, dst = argv[0], argv[1]
    from rowingdata.fit_transcode import transcode_garmin_fit_to_rowingdata
    transcode_garmin_fit_to_rowingdata(src, dst, instroke_export='full')
    print('Wrote', dst)
    return 0


if __name__ == '__main__':
    sys.exit(main())
