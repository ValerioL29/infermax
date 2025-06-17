docker build . \
    -f Dockerfile \
    --platform linux/amd64 \
    --tag registry.rcp.epfl.ch/dias-jli/infermax:v0.3 \
    --build-arg LDAP_GROUPNAME=dias \
    --build-arg LDAP_GID=11178 \
    --build-arg LDAP_USERNAME=jli \
    --build-arg LDAP_UID=271474