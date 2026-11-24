(function () {
  const asli = window.fetch.bind(window);
  let sedangKeluar = false;

  window.fetch = async function (url, opts) {
    opts = opts || {};
    const alamat = typeof url === 'string' ? url : (url && url.url) || '';
    const apiSendiri = alamat.startsWith('/api/');

    if (apiSendiri) {
      const token = localStorage.getItem('hv_token');
      if (token) {
        const h = new Headers(opts.headers || {});
        h.set('Authorization', 'Bearer ' + token);
        opts = Object.assign({}, opts, { headers: h });
      }
    }

    const res = await asli(url, opts);

    if (apiSendiri && !alamat.startsWith('/api/login') && !sedangKeluar) {
      if (res.status === 401) {
        sedangKeluar = true;
        localStorage.removeItem('hv_token');
        localStorage.removeItem('hv_user_email');
        localStorage.removeItem('hv_user_nama');
        localStorage.removeItem('hv_user_role');
        window.location.href = 'login.html';
      } else if (res.status === 403) {
        sedangKeluar = true;
        alert('Akses ditolak: fitur ini hanya untuk admin.');
        window.location.href = 'index.html';
      }
    }
    return res;
  };
})();