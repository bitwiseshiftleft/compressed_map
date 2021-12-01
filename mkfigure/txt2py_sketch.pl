#!/usr/bin/env perl

print("data = [");
while (<>) {
    if (/... actual counts\s+(\d+) \/ (\d+)\s+(\d+) \/ (\d+)/) {
        $a = $1*1.0/($2+$4);
    } elsif (/Create sketch.../) {
        $state = 1;
    } elsif (/Sketch created; sanity checking.../) {
        $state = 2;
    } elsif (/took ([0-9\.]+) seconds/ and $state==1) {
        $time = $1;
    } elsif (/took ([0-9\.]+) seconds/ and $state==2) {
        $querytime = $1;
    } elsif (/size\s+=\s+(\d+)\s+bytes/i) {
        print("($a,$time,$1,$querytime),\n");
    } else {
        $state = 0;
    }
}
print("]");
