# Legal information in Source Code Files

:zap: __This file is not supposed to be published. It is solely for internal
use!__

## Copyright header

### Bosch Code

Add a copyright / license header to all source files. Note: copyright holder is
always the legal entity. Associate can be added as author (if written agreement
is available).

```c++
// <one line to give the program's name and a brief idea of what it does.>
// Copyright (c) 2019 Robert Bosch GmbH
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

### Third-party code

If an entire source code file is from a third party, leave all
copyright/license information as is or add appropriate copyright/license
information, e.g.

```c++
// This source code is from Awesome Project V 0.9
//   (https://github.com/awesome/project/tree/v0.9
// Copyright (c) 2012-2014 Awesome Inc.
// This source code is licensed under the MIT license found in the
// 3rd-party-licenses.txt file in the root directory of this source tree.
```

I.e. treat as unmodified 3rd-party component.

## Snippet documentation

### Bosch Code with snippets

Copyright header as for Bosch Code above.

Ensure snippets are commented properly including the following information:
origin (project, version, link), copyright holder, license (alternatively: add
snippet in an atomic commit with the license information there and add required
legal information to the 3rd-party-licenses.txt)

```c++
// <one line to give the program's name and a brief idea of what it does.>
// Copyright (c) 2019 Robert Bosch GmbH
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <this_and_that>

// The following class/fct/snippet is from Awesome Project V 0.9
//   (https://github.com/awesome/project/tree/v0.9
// Copyright (c) 2012-2014 Awesome Inc., licensed under the MIT license,
// cf. 3rd-party-licenses.txt file in the root directory of this source tree.
void awesome_function() {
    return;
}

...
```

### Third-party code with Bosch snippets

As we are creating a derived work licensed under the AGPL-3.0 license,
add the usual copyright header and information on the source:

```c++
// <one line to give the program's name and a brief idea of what it does.>
// Copyright (c) 2019 Robert Bosch GmbH
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// This source code is derived from Awesome Project V 0.9
//   (https://github.com/awesome/project/tree/v0.9
// Copyright (c) 2012-2014 Awesome Inc., licensed under the MIT license,
// cf. 3rd-party-licenses.txt file in the root directory of this source tree.
```

Again: it would be best if the original version is committed as a first, then
the Bosch changes as follow-up commit.
