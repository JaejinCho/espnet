## This is to utilize git for better tracking of codes at least from this moment (as of 20210226) starting after the commit (bb05e992aa84677c882e0c444889acb9c0baef1e). Note that most of the codes in that commit were NOT written in an orginized way, most of which were generated in a messy way. ****** I need to spare some time to combine relevant codes to reduce # scripts.

### How I did add/commit/push all changes that have NOT been staged for a long time to this new branch (the commit id is as above). After the push, I start tracking changes normally with GIT at leasat from that point.
```
# step 1 (in the root directory, where .git/ resides): Get a list of all the modified/newly-added files recursively once (so still some items in the list are directories) except ones in ".gitignore"
du -sch $(for fname in `git status --porcelain | awk '{print $2}'`;do git status --porcelain $fname; done | awk '{print $2}') | sort -hk 1 > git_modified_list.all
# step 2: Remove manually the lines of file names where the size of each file > 1M
open "git_modified_list.all" generated above. From the line that changes from *K to *M, delete all the lines below and save the rest into "git_modified_list.lessthan1M"
# step 3: Get only the list
awk '{print $2}' git_modified_list.lessthan1M > git_modified_list.gitcommit
# step 4: Run git add (commit/push after this if needed)
xargs -a git_modified_list.gitcommit -d '\n' git add
# step 5: Put the list of the modified/added files > 1M into .gitignore (assuming codes usually cannot be that big. Remove some of them from the list when needed)
printf "\n# some files > 1M in clsp grid\n" >> .gitignore
git status --porcelain | grep "^??" | awk '{print $2}' >> .gitignore
```
