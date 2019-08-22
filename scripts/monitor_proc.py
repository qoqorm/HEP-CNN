#!/usr/bin/env python
import sys, os, time
import subprocess
import csv

wait_second = 10
procId = int(sys.argv[1])

pagesize = int(subprocess.check_output(['getconf', 'PAGESIZE']))
nproc = int(subprocess.check_output(['nproc', '--all']))

procdir = '/proc/%d' % procId
columns = ["cpu", "mem", "io_read", "io_write"]
with open('proc_%d.csv' % procId, 'w') as fout:
  writer = csv.writer(fout)
  writer.writerow(columns)

  utime0, stime0, rss = 0, 0, 0
  io_read0, io_write0 = 0, 0
  totaltime0 = 0
  while True:
      if not os.path.exists(procdir): break
      utime1, stime1, rss1 = 0, 0, 0
      io_read1, io_write1 = 0, 0
      totaltime1 = 0

      stat = [0]*len(columns)
      with open(procdir+'/stat') as f:
          ## Note: see the linux man page proc(5)
          statcpu = f.read().split()
          utime1 = int(statcpu[13]) ## utime in jiffies unit
          stime1 = int(statcpu[14]) ## utime in jiffies unit
          rss1 = int(statcpu[23]) ## rss

      with open(procdir+'/io') as f:
          ## Note: /proc/PROC/io gives [rchar, wchar, syscr, syscw, read_bytes, write_bytes, cancelled_write_bytes]
          statio = f.readlines()
          io_read1 = int(statio[4].rsplit(':', 1)[-1])
          io_write1 = int(statio[5].rsplit(':', 1)[-1])
      with open('/proc/stat') as f:
          stattotal = f.readlines()[0].split()
          totaltime1 = sum(int(x) for x in stattotal[1:]) ## cpu time in jiffies unit

      if totaltime0 != 0:
          cpu_usage = 100*nproc*float( (utime1-utime0) + (stime1-stime0) ) / (totaltime1-totaltime0)
          io_read = io_read1-io_read0
          io_write = io_write1-io_write0

          print(cpu_usage, rss1, io_read, io_write)
          writer.writerow(stat)

      utime0, stime0 = utime1, stime1
      io_read0, io_write0 = io_read1, io_write1
      totaltime0 = totaltime1

      time.sleep(wait_second)

