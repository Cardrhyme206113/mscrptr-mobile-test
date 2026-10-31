#!/usr/bin/env bash
set -euo pipefail
adb install -r pocket-native/app/build/outputs/apk/debug/app-debug.apk
adb logcat -c
adb shell am start -W -n com.muscript.pocket/.MainActivity
sleep 12
adb logcat -d -v time > startup-logcat.txt
if grep -q 'Process: com.muscript.pocket' startup-logcat.txt; then
  echo 'APP CRASHED DURING LAUNCH:'
  grep -A 33 -B 4 'Process: com.muscript.pocket' startup-logcat.txt | tail -150
  exit 1
fi
if ! adb shell pidof com.muscript.pocket >/dev/null; then
  echo 'APP PROCESS NOT RUNNING AFTER LAUNCH'
  exit 1
fi
echo 'PASS: App stayed alive 12 seconds and no AndroidRuntime fatal exception logged'
