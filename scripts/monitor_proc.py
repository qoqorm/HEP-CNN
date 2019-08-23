#!/usr/bin/env python
import sys, os, time
import subprocess
import csv

class SysStat:
    def __init__(self, procId, verbose=False):
        self.verbose = verbose
        self.procdir = '/proc/%d' % procId

        #self.pagesize = int(subprocess.check_output(['getconf', 'PAGESIZE']))
        self.nproc = int(subprocess.check_output(['nproc', '--all']))

        self.utime, self.stime, self.rss = 0, 0, 0
        self.io_read, self.io_write = 0, 0
        self.totaltime = 0

        self.cpuFracs = []
        self.readBytes = []
        self.writeBytes = []

        self.writer = csv.writer(open('proc_%d.csv' % procId, 'w'))
        columns = ["CPU", "RSS", "Read_Bytes", "Write_Bytes", "Annotation"]
        self.writer.writerow(columns)

        self.update()

    def update(self, annotation=None):
        if not os.path.exists(self.procdir): return False
        if annotation == None: "''"

        with open(self.procdir+'/stat') as f:
            ## Note: see the linux man page proc(5)
            statcpu = f.read().split()
            utime = int(statcpu[13]) ## utime in jiffies unit
            stime = int(statcpu[14]) ## utime in jiffies unit
            rss   = int(statcpu[23]) ## rss

        with open(self.procdir+'/io') as f:
            ## Note: /proc/PROC/io gives [rchar, wchar, syscr, syscw, read_bytes, write_bytes, cancelled_write_bytes]
            statio = f.readlines()
            io_read  = int(statio[4].rsplit(':', 1)[-1])
            io_write = int(statio[5].rsplit(':', 1)[-1])

        with open('/proc/stat') as f:
            stattotal = f.readlines()[0].split()
            totaltime = sum(int(x) for x in stattotal[1:]) ## cpu time in jiffies unit

        if self.totaltime != 0:
            cpuFrac   = 100*self.nproc*float( (utime-self.utime) + (stime-self.stime) ) / (totaltime-self.totaltime)
            readByte  = io_read-self.io_read
            writeByte = io_write-self.io_write

            self.cpuFracs.append(cpuFrac)
            self.readBytes.append(readByte)
            self.writeBytes.append(writeByte)

            stat = [cpuFrac, rss, readByte, writeByte, annotation]
            if self.verbose: print(stat)
            self.writer.writerow(stat)

        self.utime, self.stime, self.rss = utime, stime, rss
        self.io_read, self.io_write = io_read, io_write
        self.totaltime = totaltime

        return True

if __name__ == '__main__':
    wait_second = 10
    procId = int(sys.argv[1])

    sysstat = SysStat(procId, verbose=True)
    while sysstat.update(): time.sleep(wait_second)

