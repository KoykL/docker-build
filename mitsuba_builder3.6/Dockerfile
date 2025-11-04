FROM quay.io/pypa/manylinux_2_28_x86_64 AS builder

# Make sure pip/setuptools/wheel are up to date inside the installed interpreter
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel ninja virtualenv

# Optional: create a venv template (uncomment if desired)
# RUN ${PREFIX}/bin/python3.10 -m venv ${PREFIX}/venv

RUN dnf update && dnf install -y ccache

RUN python3.10 -m pip install --no-cache-dir scikit-build scikit-build-core pybind11 build typing-extensions hatch_fancy_pypi_readme cmake==3.31.6
ARG PLAT=manylinux_2_28_x86_64
COPY . /mitsuba
WORKDIR /mitsuba
ENV CCACHE_DIR=/ccache

RUN --mount=type=cache,target=/ccache cd ext/nanobind && rm -rf dist && CMAKE_ARGS="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache" python3.10 -m pip install .
RUN --mount=type=cache,target=/ccache cd ext/drjit && rm -rf dist  && CMAKE_ARGS="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache" python3.10 -m pip wheel -v --no-build-isolation -w build_whl . &&  auditwheel -vv show build_whl/drjit*.whl && auditwheel repair build_whl/drjit*.whl --plat "$PLAT" -w build_whl &&  python3.10 -m pip install ./build_whl/typing_extensions*.whl ./build_whl/drjit*manylinux*.whl;
#RUN cd ext/drjit && auditwheel show dist/drjit-1.0.3-cp310-cp310-linux_x86_64.whl

#&& auditwheel repair dist/*.whl --plat "$PLAT" -w output && python3.10 -m pip install ./dist/*.whl

# RUN --mount=type=cache,target=/ccache rm -rf dist && CMAKE_CXX_COMPILER_LAUNCHER=ccache python3 -m build --wheel
# RUN  --mount=type=cache,target=/ccache mkdir build && cd build && CMAKE_CXX_COMPILER_LAUNCHER=ccache  cmake -G Ninja ..  && ninja -v
RUN --mount=type=cache,target=/ccache CMAKE_ARGS="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache "   python3.10 -m pip wheel -v -w build_whl --no-build-isolation . && \
  for wheel in $(ls build_whl/mitsuba*.whl); do \
    new_whl_name=$(python3.10 -m wheel tags --platform-tag manylinux_2_27_x86_64.manylinux_2_28_x86_64 ${wheel} | tail -1 | cat); \
  done;
FROM scratch AS wheels
COPY --from=builder /mitsuba/build_whl/mitsuba*manylinux*.whl /
COPY --from=builder /mitsuba/ext/drjit/build_whl/*manylinux*.whl /
# COPY --from=builder /mitsuba/build_whl /