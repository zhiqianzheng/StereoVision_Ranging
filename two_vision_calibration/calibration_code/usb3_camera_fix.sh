#!/bin/bash
echo 'on' > /sys/bus/usb/devices/*/power/level
for i in /sys/bus/usb/devices/*/power/autosuspend_delay_ms; do
    echo -1 > $i 2>/dev/null || true
done
modprobe uvcvideo quirks=0x80
