FROM fedora:41
RUN dnf install -y git
RUN useradd -u 1000 -G wheel user
RUN echo '%wheel	ALL=(ALL)	NOPASSWD: ALL' >> /etc/sudoers
USER user

WORKDIR /home/user