#!/usr/bin/env perl

print("data = [");
while (<>) {
    print if s/
        Size    \s* (\d+ \s*\*\s* \d+\*\d+ \s*-\s* \d+).*
        pass.rate  \s*=\s* (\d+) \s*\/\s* (\d+).*
        rand.trial \s*=\s* ([0-9\.]+) .*
        sample.row \s*=\s* ([0-9\.]+) .*
        build.row  \s*=\s* ([0-9\.]+) .*
        query.row  \s*=\s* ([0-9\.]+) .*
    /
        "(" . eval($1) . ", $2\/$3., $4, $5, $6, $7),"
    /exg;
}
print("]\n");
