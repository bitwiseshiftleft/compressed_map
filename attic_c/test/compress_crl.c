/** @file lfr_file.c
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Demo to compress CRLs.
 * Based on https://fm4dd.com/openssl/crldisplay.shtm
 * FIXME: remove all the original code.
 */

#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/x509.h>
#include "lfr_nonuniform.h"
#include <stdio.h>
#include "assert.h"

int main() {
    const char *crl_filestr = "test/wwdrca.crl";

    /* Initialize */
    OpenSSL_add_all_algorithms();
    ERR_load_BIO_strings();

    /* Open CRL file */
    BIO *crlbio = BIO_new(BIO_s_file());
    int ret = BIO_read_filename(crlbio, crl_filestr);
    if (ret <= 0) {
        printf("Error loading file %s: %d\n", crl_filestr, ret);
        return ret;
    }
    X509_CRL *mycrl = d2i_X509_CRL_bio(crlbio, NULL);
    if (mycrl == NULL) {
        printf("Failed to load CRL\n");
        return 0; // TODO: cleanup
    }
    STACK_OF(X509_REVOKED) *rev = X509_CRL_get_REVOKED(mycrl);
    ssize_t revoked = sk_X509_REVOKED_num(rev);
    printf("Found %ld revoked certs\n", revoked);

    lfr_builder_t builder;
    // not big enough; eh whatever it'll get bigger
    ret = lfr_builder_init(builder,1000,1000,0);
    if (ret) {
        fprintf(stderr,"Error %d\n",ret);
        return ret;
    }

    uint8_t *buffer = NULL;
    ssize_t len_buffer = 0;

    lfr_response_t response = 1; // intentional warning
    for (ssize_t i = 0; i < revoked; i++) {
        if (i % 100000 == 0) printf("Process cert number %ld / %ld\n", i, revoked);
        const X509_REVOKED *rev_entry = sk_X509_REVOKED_value(rev, i);
        const ASN1_INTEGER *serial = X509_REVOKED_get0_serialNumber(rev_entry);
        /* TODO: why isn't this method const? */
        int len = i2d_ASN1_INTEGER((ASN1_INTEGER*)serial, NULL); 

        if (len < 0) {
            printf("Len was %d!\n", len);
            // TODO
            break;
        } else if (len > len_buffer) {
            buffer = realloc(buffer, 2*len);
            if (buffer == NULL) {
                printf("Failed to reallocate as %d bytes\n", 2*len);
                return -1;
            }
        }

        uint8_t *buffer_tmp = buffer;
        /* TODO: why isn't it const? */
        int len1 = i2d_ASN1_INTEGER((ASN1_INTEGER*)serial, &buffer_tmp); 
        assert(len == len1 && buffer_tmp == buffer+len);
        ret = lfr_builder_insert(builder, buffer, buffer_tmp-buffer, i%2); // TODO: how to get non-revoked certs?
        if (ret) {
            printf("Insert error %d\n", ret);
            break;
        }
    }

    printf("Build ...\n");
    lfr_nonuniform_map_t map;
    ret = lfr_nonuniform_build(map, builder);
    if (ret) {
        printf("Build error %d\n", ret);
    } else {
        printf("Success!  Size=%ld\n", lfr_nonuniform_map_serial_size(map));
    }

    /* TODO: serialize */

    lfr_nonuniform_map_destroy(map);
    lfr_builder_destroy(builder);

    X509_CRL_free(mycrl);
    BIO_free_all(crlbio);
    exit(0);
}
