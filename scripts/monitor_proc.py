#!/usr/bin/env python
import sys, os, time
import subprocess
import csv

procId = int(sys.argv[1])
pagesize = int(subprocess.check_output(['getconf', 'PAGESIZE']))

procdir = '/proc/%d' % procId
columns = ["cpu", "mem", "io_read", "io_write"]
with open('proc_%d.csv' % procId, 'w') as fout:
  writer = csv.writer(fout)
  writer.writerow(columns)
  while True:
      if not os.path.exists(procdir): break
      stat = [0]*len(columns)
      with open(procdir+'/stat') as fcpu:
          ## Note: see the linux man page proc(5)
          statcpu = fcpu.read().split()
          stat[0] = float(statcpu[13]) ## utime
          stat[1] = float(statcpu[23]) ## rss
      with open(procdir+'/io') as fio:
          ## Note: /proc/PROC/io gives [rchar, wchar, syscr, syscw, read_bytes, write_bytes, cancelled_write_bytes]
          statio = fio.readlines()
          io_read = int(statio[4].rsplit(':', 1)[-1])
          io_write = int(statio[5].rsplit(':', 1)[-1])
          stat[2] = io_read
          stat[3] = io_write

      print(stat)
      writer.writerow(stat)
      time.sleep(1)


