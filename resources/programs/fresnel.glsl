float Fresnel (float th, float n) {
    float cosi = cos (th);
    float R = 1.0f;
    float n12 = 1.0f / n;
    float sint = n12 * sqrt (1 - (cosi * cosi));

    if (sint < 1.0f) {
        float cost = sqrt (1.0 - (sint * sint));
        float r_ortho = (cosi - n * cost) / (cosi + n * cost);
        float r_par = (cost - n * cosi) / (cost + n * cosi);
        R = (r_ortho * r_ortho + r_par * r_par) / 2;
    }
    return R;
}
